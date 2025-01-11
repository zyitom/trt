/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <algorithm>
#include <array>
#include <chrono>
#include <cuda_profiler_api.h>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#if defined(__QNX__)
#include <sys/neutrino.h>
#include <sys/syspage.h>
#endif

#include "NvInfer.h"

#include "ErrorRecorder.h"
#include "logger.h"
#include "sampleDevice.h"
#include "sampleEngines.h"
#include "sampleInference.h"
#include "sampleOptions.h"
#include "sampleUtils.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "preprocess.cuh"
using namespace nvinfer1;
namespace sample
{

template <class TMapType, class TEngineType>
bool validateTensorNames(TMapType const& map, TEngineType const* engine, int32_t const endBindingIndex)
{
    // Check if the provided input tensor names match the input tensors of the engine.
    // Throw an error if the provided input tensor names cannot be found because it implies a potential typo.
    for (auto const& item : map)
    {
        bool tensorNameFound{false};
        for (int32_t b = 0; b < endBindingIndex; ++b)
        {
            auto const tensorName = engine->getIOTensorName(b);
            auto const tensorIOMode = engine->getTensorIOMode(tensorName);
            if (tensorIOMode == nvinfer1::TensorIOMode::kINPUT && matchStringWithOneWildcard(item.first, tensorName))
            {
                tensorNameFound = true;
                break;
            }
        }
        if (!tensorNameFound)
        {
            sample::gLogError << "Cannot find input tensor with name \"" << item.first << "\" in the engine bindings! "
                              << "Please make sure the input tensor names are correct." << std::endl;
            return false;
        }
    }
    return true;
}

template <class TEngineType>
class FillBindingClosure
{
private:
    using InputsMap = std::unordered_map<std::string, std::string>;
    using BindingsVector = std::vector<std::unique_ptr<Bindings>>;

    TEngineType const* mEngine;
    nvinfer1::IExecutionContext const* mContext;
    InputsMap const& inputs;
    BindingsVector& bindings;
    int32_t batch;
    int32_t endBindingIndex;
    int32_t profileIndex;

    void fillOneBinding(TensorInfo const& tensorInfo)
    {
        auto const name = tensorInfo.name;
        auto const* bindingInOutStr = tensorInfo.isInput ? "Input" : "Output";
        for (auto& binding : bindings)
        {
            auto const input = findPlausible(inputs, name);
            if (tensorInfo.isInput && input != inputs.end())
            {
                sample::gLogInfo << "Using values loaded from " << input->second << " for input " << name << std::endl;
                binding->addBinding(tensorInfo, input->second);
            }
            else
            {
                if (tensorInfo.isInput)
                {
                    sample::gLogInfo << "Using random values for input " << name << std::endl;
                }
                binding->addBinding(tensorInfo);
            }

                sample::gLogInfo << bindingInOutStr << " binding for " << name << " with dimensions " << tensorInfo.dims
                                 << " is created." << std::endl;
            
        }
    }

    bool fillAllBindings(int32_t batch, int32_t endBindingIndex)
    {
        if (!validateTensorNames(inputs, mEngine, endBindingIndex))
        {
            sample::gLogError << "Invalid tensor names found in --loadInputs flag." << std::endl;
            return false;
        }
        for (int32_t b = 0; b < endBindingIndex; b++)
        {
            TensorInfo tensorInfo;
            tensorInfo.bindingIndex = b;
            getTensorInfo(tensorInfo);
            tensorInfo.updateVolume(batch);
            fillOneBinding(tensorInfo);
        }
        return true;
    }

    void getTensorInfo(TensorInfo& tensorInfo);

public:
    FillBindingClosure(TEngineType const* _engine, nvinfer1::IExecutionContext const* _context,
        InputsMap const& _inputs, BindingsVector& _bindings, int32_t _batch, int32_t _endBindingIndex,
        int32_t _profileIndex)
        : mEngine(_engine)
        , mContext(_context)
        , inputs(_inputs)
        , bindings(_bindings)
        , batch(_batch)
        , endBindingIndex(_endBindingIndex)
        , profileIndex(_profileIndex)
    {
    }

    bool operator()()
    {
        return fillAllBindings(batch, endBindingIndex);
    }
};

template <>
void FillBindingClosure<nvinfer1::ICudaEngine>::getTensorInfo(TensorInfo& tensorInfo)
{
    auto const b = tensorInfo.bindingIndex;
    auto const name = mEngine->getIOTensorName(b);
    tensorInfo.name = name;
    tensorInfo.dims = mContext->getTensorShape(name);
    tensorInfo.comps = mEngine->getTensorComponentsPerElement(name, profileIndex);
    tensorInfo.strides = mContext->getTensorStrides(name);
    tensorInfo.vectorDimIndex = mEngine->getTensorVectorizedDim(name, profileIndex);
    tensorInfo.isInput = mEngine->getTensorIOMode(name) == TensorIOMode::kINPUT;
    tensorInfo.dataType = mEngine->getTensorDataType(name);
}

namespace
{
bool allocateContextMemory(InferenceEnvironment& iEnv, InferenceOptions const& inference)
{
    auto* engine = iEnv.engine.get();
    iEnv.deviceMemory.resize(inference.infStreams);
    // Delay context memory allocation until input shapes are specified because runtime allocation would require actual
    // input shapes.
    for (int32_t i = 0; i < inference.infStreams; ++i)
    {
        auto const& ec = iEnv.contexts.at(i);
        if (inference.memoryAllocationStrategy == MemoryAllocationStrategy::kSTATIC)
        {
            sample::gLogInfo << "Created execution context with device memory size: "
                             << (engine->getDeviceMemorySize() / 1.0_MiB) << " MiB" << std::endl;
        }
        // else
        // {
        //     size_t sizeToAlloc{0};
        //     const char* allocReason{nullptr};
        //     if (inference.memoryAllocationStrategy == MemoryAllocationStrategy::kPROFILE)
        //     {
        //         auto const p = inference.optProfileIndex;
        //         sizeToAlloc = engine->getDeviceMemorySizeForProfile(p);
        //         allocReason = "current profile";
        //     }
        //     else if (inference.memoryAllocationStrategy == MemoryAllocationStrategy::kRUNTIME)
        //     {
        //         sizeToAlloc = ec->updateDeviceMemorySizeForShapes();
        //         allocReason = "current input shapes";
        //     }
        //     else
        //     {
        //         sample::gLogError << "Unrecognizable memory allocation strategy." << std::endl;
        //         return false;
        //     }
        //     iEnv.deviceMemory.at(i) = TrtDeviceBuffer(sizeToAlloc);
        //     ec->setDeviceMemoryV2(iEnv.deviceMemory.at(i).get(), iEnv.deviceMemory.at(i).getSize());
        //     sample::gLogInfo << "Maximum device memory size across all profiles: "
        //                      << (engine->getDeviceMemorySizeV2() / 1.0_MiB) << " MiB" << std::endl;
        //     sample::gLogInfo << "Only allocated device memory enough for " << allocReason << ": "
        //                      << (sizeToAlloc / 1.0_MiB) << " MiB" << std::endl;
        // }
    }
    return true;
}
} // namespace

bool setUpInference(InferenceEnvironment& iEnv, InferenceOptions const& inference, SystemOptions const& system)
{

    int32_t device{};
    CHECK(cudaGetDevice(&device));

    cudaDeviceProp properties;
    CHECK(cudaGetDeviceProperties(&properties, device));
    int32_t const isIntegrated{properties.integrated};
    // Use managed memory on integrated devices when transfers are skipped
    // and when it is explicitly requested on the commandline.
    bool useManagedMemory{(inference.skipTransfers && isIntegrated) || inference.useManaged};
    SMP_RETVAL_IF_FALSE(!iEnv.safe, "Safe inference is not supported!", false, sample::gLogError);

    using FillStdBindings = FillBindingClosure<nvinfer1::ICudaEngine>;

    auto* engine = iEnv.engine.get();
    SMP_RETVAL_IF_FALSE(engine != nullptr, "Got invalid engine!", false, sample::gLogError);

    // Release serialized blob to save memory space.
    iEnv.engine.releaseBlob();

    // Setup weight streaming if enabled
    if (engine->getStreamableWeightsSize() > 0)
    {
        auto const& budget = inference.weightStreamingBudget;
        int64_t wsBudget = budget.bytes;
        if (budget.percent != 100.0)
        {
            double const percent = budget.percent;
            ASSERT(percent < 100.0);
            auto const max = engine->getStreamableWeightsSize();
            wsBudget = (max >= 0) ? (percent / 100) * (max) : WeightStreamingBudget::kDISABLE;
        }

        if (wsBudget == WeightStreamingBudget::kDISABLE)
        {
            wsBudget = engine->getStreamableWeightsSize();
        }
        else if (wsBudget == WeightStreamingBudget::kAUTOMATIC)
        {
            wsBudget = engine->getWeightStreamingAutomaticBudget();
        }
        ASSERT(wsBudget >= 0);
        bool success = engine->setWeightStreamingBudgetV2(wsBudget);
        SMP_RETVAL_IF_FALSE(success, "Failed to set weight streaming limit!", false, sample::gLogError);
        switch (wsBudget)
        {
        case WeightStreamingBudget::kDISABLE:
        {
            sample::gLogInfo << "Weight streaming has been disabled at runtime." << std::endl;
            break;
        }

        case WeightStreamingBudget::kAUTOMATIC:
        {
            sample::gLogInfo << "The weight streaming budget will automatically be chosen by TensorRT." << std::endl;
            break;
        }
        default:
        {
            sample::gLogInfo << "Weight streaming is enabled with a device memory limit of " << wsBudget << " bytes."
                             << std::endl;
            break;
        }
        }
    }

    int32_t const nbOptProfiles = engine->getNbOptimizationProfiles();

    if (inference.optProfileIndex >= nbOptProfiles)
    {
        sample::gLogError << "Selected profile index " << inference.optProfileIndex
                          << " exceeds the number of profiles that the engine holds. " << std::endl;
        return false;
    }

    if (nbOptProfiles > 1 && !inference.setOptProfile)
    {
        sample::gLogWarning << nbOptProfiles
                            << " profiles detected but not set. Running with profile 0. Please use "
                               "--dumpOptimizationProfile to see all available profiles."
                            << std::endl;
    }

    cudaStream_t setOptProfileStream;
    CHECK(cudaStreamCreate(&setOptProfileStream));

    for (int32_t s = 0; s < inference.infStreams; ++s)
    {
        IExecutionContext* ec{nullptr};
        if (inference.memoryAllocationStrategy == MemoryAllocationStrategy::kSTATIC)
        {
            // Let TRT pre-allocate and manage the memory.
            ec = engine->createExecutionContext();
        }
        else
        {
            // Allocate based on the current profile or runtime shapes.
            ec = engine->createExecutionContext(ExecutionContextAllocationStrategy::kUSER_MANAGED);
        }
        if (ec == nullptr)
        {
            sample::gLogError << "Unable to create execution context for stream " << s << "." << std::endl;
            return false;
        }
        ec->setNvtxVerbosity(inference.nvtxVerbosity);


        int32_t const persistentCacheLimit
            = samplesCommon::getMaxPersistentCacheSize() * inference.persistentCacheRatio;
        sample::gLogInfo << "Setting persistentCacheLimit to " << persistentCacheLimit << " bytes." << std::endl;
        ec->setPersistentCacheLimit(persistentCacheLimit);

        auto setProfile = ec->setOptimizationProfileAsync(inference.optProfileIndex, setOptProfileStream);
        CHECK(cudaStreamSynchronize(setOptProfileStream));

        if (!setProfile)
        {
            sample::gLogError << "Set optimization profile failed. " << std::endl;
            if (inference.infStreams > 1)
            {
                sample::gLogError
                    << "Please ensure that the engine is built with preview feature profileSharing0806 enabled. "
                    << std::endl;
            }
            return false;
        }

        iEnv.contexts.emplace_back(ec);
        iEnv.bindings.emplace_back(new Bindings(useManagedMemory));
    }

    CHECK(cudaStreamDestroy(setOptProfileStream));



    int32_t const endBindingIndex = engine->getNbIOTensors();

    // Make sure that the tensor names provided in command-line args actually exist in any of the engine bindings
    // to avoid silent typos.
    if (!validateTensorNames(inference.shapes, engine, endBindingIndex))
    {
        sample::gLogError << "Invalid tensor names found in --shapes flag." << std::endl;
        return false;
    }

    for (int32_t b = 0; b < endBindingIndex; ++b)
    {
        auto const& name = engine->getIOTensorName(b);
        auto const& mode = engine->getTensorIOMode(name);
        if (mode == TensorIOMode::kINPUT)
        {
            Dims const dims = iEnv.contexts.front()->getTensorShape(name);
            bool isShapeInferenceIO{false};
            isShapeInferenceIO = engine->isShapeInferenceIO(name);
            bool const hasRuntimeDim = std::any_of(dims.d, dims.d + dims.nbDims, [](int32_t dim) { return dim == -1; });
            auto const shape = findPlausible(inference.shapes, name);
            if (hasRuntimeDim || isShapeInferenceIO)
            {
                // Set shapeData to either dimensions of the input (if it has a dynamic shape)
                // or set to values of the input (if it is an input shape tensor).
                std::vector<int32_t> shapeData;

                if (shape == inference.shapes.end())
                {
                    // No information provided. Use default value for missing data.
                    constexpr int32_t kDEFAULT_VALUE = 1;
                    if (isShapeInferenceIO)
                    {
                        // Set shape tensor to all ones.
                        shapeData.assign(volume(dims, 0, dims.nbDims), kDEFAULT_VALUE);
                        sample::gLogWarning << "Values missing for input shape tensor: " << name
                                            << "Automatically setting values to: " << shapeData << std::endl;
                    }
                    else
                    {
                        // Use default value for unspecified runtime dimensions.
                        shapeData.resize(dims.nbDims);
                        std::transform(dims.d, dims.d + dims.nbDims, shapeData.begin(),
                            [&](int32_t dimension) { return dimension >= 0 ? dimension : kDEFAULT_VALUE; });
                        sample::gLogWarning << "Shape missing for input with dynamic shape: " << name
                                            << "Automatically setting shape to: " << shapeData << std::endl;
                    }
                }
                else if (inference.inputs.count(shape->first) && isShapeInferenceIO)
                {
                    // Load shape tensor from file.
                    int64_t const size = volume(dims, 0, dims.nbDims);
                    shapeData.resize(size);
                    auto const& filename = inference.inputs.at(shape->first);
                    auto dst = reinterpret_cast<char*>(shapeData.data());
                    loadFromFile(filename, dst, size * sizeof(decltype(shapeData)::value_type));
                }
                else
                {
                    shapeData = shape->second;
                }

                int32_t* shapeTensorData{nullptr};
                if (isShapeInferenceIO)
                {
                    // Save the data in iEnv, in a way that it's address does not change
                    // before enqueueV3 is called.
                    iEnv.inputShapeTensorValues.emplace_back(shapeData);
                    shapeTensorData = iEnv.inputShapeTensorValues.back().data();
                }

                for (auto& c : iEnv.contexts)
                {
                    if (isShapeInferenceIO)
                    {
                        sample::gLogInfo << "Set input shape tensor " << name << " to: " << shapeData << std::endl;
                        if (!c->setTensorAddress(name, shapeTensorData))
                        {
                            return false;
                        }
                    }
                    else
                    {
                        sample::gLogInfo << "Set shape of input tensor " << name << " to: " << shapeData
                                            << std::endl;
                        if (!c->setInputShape(name, toDims(shapeData)))
                        {
                            return false;
                        }
                    }
                }
            }
            else if (nbOptProfiles && shape != inference.shapes.end())
            {
                // Check if the provided shape matches the static dimensions in the engine.
                for (auto& c : iEnv.contexts)
                {
                    if (!c->setInputShape(name, toDims(shape->second)))
                    {
                        sample::gLogError << "The engine was built with static shapes for input tensor " << name
                                          << " but the provided shapes do not match the static shapes!" << std::endl;
                        return false;
                    }
                }
            }
        }
    }


    if (!allocateContextMemory(iEnv, inference))
    {
        return false;
    }

    auto const* context = iEnv.contexts.front().get();
    bool fillBindingsSuccess = FillStdBindings(
        engine, context, inference.inputs, iEnv.bindings, 1, endBindingIndex, inference.optProfileIndex)();



    return fillBindingsSuccess;
}

TaskInferenceEnvironment::TaskInferenceEnvironment(
    std::string engineFile, InferenceOptions inference, int32_t deviceId, int32_t DLACore, int32_t bs)
    : iOptions(inference)
    , device(deviceId)
    , batch(bs)
{
    BuildEnvironment bEnv(/* isSafe */ false, /* versionCompatible */ false, DLACore, "", getTempfileControlDefaults());
    loadEngineToBuildEnv(engineFile, bEnv, sample::gLogError);
    std::unique_ptr<InferenceEnvironment> tmp(new InferenceEnvironment(bEnv));
    iEnv = std::move(tmp);

    CHECK(cudaSetDevice(device));
    SystemOptions system{};
    system.device = device;
    system.DLACore = DLACore;
    if (!setUpInference(*iEnv, iOptions, system))
    {
        sample::gLogError << "Inference set up failed" << std::endl;
    }
}


    void Binding::dump(std::ostream& os, Dims dims, Dims strides, int32_t vectorDim, int32_t spv,
        std::string const separator /*= " "*/) const
{
    void* outputBuffer{};
    if (outputAllocator != nullptr)
    {
        outputBuffer = outputAllocator->getBuffer()->getHostBuffer();
        // Overwrite dimensions with those reported by the output allocator.
        dims = outputAllocator->getFinalDims();
        os << "Final shape is " << dims << " reported by the output allocator." << std::endl;
    }
    else
    {
        outputBuffer = buffer->getHostBuffer();
    }
    switch (dataType)
    {
        case nvinfer1::DataType::kFLOAT:
        {
            dumpBuffer<float>(outputBuffer, separator, os, dims, strides, vectorDim, spv);
            break;
        }
    }
}

namespace
{


// using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;



//!
//! \struct SyncStruct
//! \brief Threads synchronization structure
//!
struct SyncStruct
{
    std::mutex mutex;
    TrtCudaStream mainStream;
    TrtCudaEvent gpuStart{cudaEventBlockingSync};
};

struct Enqueue
{
    explicit Enqueue(nvinfer1::IExecutionContext& context)
        : mContext(context)
    {
    }

    nvinfer1::IExecutionContext& mContext;
};

//!
//! \class EnqueueExplicit
//! \brief Functor to enqueue inference with explict batch
//!
class EnqueueExplicit : private Enqueue
{

public:
    explicit EnqueueExplicit(nvinfer1::IExecutionContext& context, Bindings const& bindings)
        : Enqueue(context)
        , mBindings(bindings)
    {
        ASSERT(mBindings.setTensorAddresses(mContext));
    }

    bool operator()(TrtCudaStream& stream) const
    {
      try {
            return mContext.enqueueV3(stream.get());
        }
        catch (const std::exception&) {
            return false;
        }
    }

private:
    Bindings const& mBindings;
};

//!
//! \class EnqueueGraph
//! \brief Functor to enqueue inference from CUDA Graph
//!
class EnqueueGraph
{

public:
    explicit EnqueueGraph(nvinfer1::IExecutionContext& context, TrtCudaGraph& graph)
        : mGraph(graph)
        , mContext(context)
    {
    }

    bool operator()(TrtCudaStream& stream) const
    {
        return mGraph.launch(stream);
    }

    TrtCudaGraph& mGraph;
    nvinfer1::IExecutionContext& mContext;
};

//!
//! \class EnqueueGraphSafe
//! \brief Functor to enqueue inference from CUDA Graph
//!
class EnqueueGraphSafe
{

public:
    explicit EnqueueGraphSafe(TrtCudaGraph& graph)
        : mGraph(graph)
    {
    }

    bool operator()(TrtCudaStream& stream) const
    {
        return mGraph.launch(stream);
    }

    TrtCudaGraph& mGraph;
};

using EnqueueFunction = std::function<bool(TrtCudaStream&)>;

enum class StreamType : int32_t
{
    kINPUT = 0,
    kCOMPUTE = 1,
    kOUTPUT = 2,
    kNUM = 3
};

enum class EventType : int32_t
{
    kINPUT_S = 0,
    kINPUT_E = 1,
    kCOMPUTE_S = 2,
    kCOMPUTE_E = 3,
    kOUTPUT_S = 4,
    kOUTPUT_E = 5,
    kNUM = 6
};

using MultiStream = std::array<TrtCudaStream, static_cast<int32_t>(StreamType::kNUM)>;

using MultiEvent = std::array<std::unique_ptr<TrtCudaEvent>, static_cast<int32_t>(EventType::kNUM)>;



//!
//! \class Iteration
//! \brief Inference iteration and streams management
//!
class Iteration
{

public:
    Iteration(int32_t id, InferenceOptions const& inference, nvinfer1::IExecutionContext& context, Bindings& bindings)
        : mBindings(bindings)
        , mStreamId(id)
        , mDepth(1 + inference.overlap)
        , mActive(mDepth)
        , mEvents(mDepth)
        , mContext(&context)
    {
        for (int32_t d = 0; d < mDepth; ++d)
        {
            for (int32_t e = 0; e < static_cast<int32_t>(EventType::kNUM); ++e)
            {
                mEvents[d][e].reset(new TrtCudaEvent(!inference.spin));
            }
        }
        createEnqueueFunction(inference, context, bindings);
    }
    // jetson不用 skiptransfer
    bool query(bool skipTransfers)
    {
        sample::gLogInfo << "Querying stream with skipTransfers: " << skipTransfers << std::endl;
        if (mActive[mNext])
        {
            return true;
        }

        if (!skipTransfers)
        {
            setInputData(false);
            wait(EventType::kINPUT_E, StreamType::kCOMPUTE); // Wait for input DMA before compute
        }

        if (!mEnqueue(getStream(StreamType::kCOMPUTE)))
        {
            return false;
        }


        if (!skipTransfers)
        {
            wait(EventType::kCOMPUTE_E, StreamType::kOUTPUT); // Wait for compute before output DMA
            fetchOutputData(false);
        }

        mActive[mNext] = true;
        moveNext();
        return true;
    }

    bool sync(bool skipTransfers) {
    if (mActive[mNext]) {
        
        if (skipTransfers) {
            getEvent(EventType::kCOMPUTE_E).synchronize();
        } else {
            getEvent(EventType::kOUTPUT_E).synchronize();
        }
        mActive[mNext] = false;
        return true;
    }
    return false;
}


    void wait(TrtCudaEvent& gpuStart)
    {
        getStream(StreamType::kINPUT).wait(gpuStart);
    }

    void setInputData(bool sync)
    {


        sample::gLogInfo << "Setting input data, sync: " << sync << std::endl;
    

        if (mBindings.imageInfo.data) {
            sample::gLogInfo << "Found image data to process" << std::endl;
        }
        mBindings.transferInputToDevice(getStream(StreamType::kINPUT));
        // additional sync to avoid overlapping with inference execution.
        if (sync)
        {
            getStream(StreamType::kINPUT).synchronize();
        }
    }

    void fetchOutputData(bool sync)
    {
        mBindings.transferOutputToHost(getStream(StreamType::kOUTPUT));
        // additional sync to avoid overlapping with inference execution.
        if (sync)
        {
            getStream(StreamType::kOUTPUT).synchronize();
        }
    }

private:

    void moveNext()
    {
        mNext = mDepth - 1 - mNext;
    }

    TrtCudaStream& getStream(StreamType t)
    {   
        return mStream[static_cast<int32_t>(t)];
    }

    TrtCudaEvent& getEvent(EventType t)
    {
        return *mEvents[mNext][static_cast<int32_t>(t)];
    }

    void record(EventType e, StreamType s)
    {
        getEvent(e).record(getStream(s));
    }

    void wait(EventType e, StreamType s)
    {
        getStream(s).wait(getEvent(e));
    }


    void createEnqueueFunction(
        InferenceOptions const& inference, nvinfer1::IExecutionContext& context, Bindings& bindings)
    {
        mEnqueue = EnqueueFunction(EnqueueExplicit(context, mBindings));
        if (inference.graph)
        {
            sample::gLogInfo << "Capturing CUDA graph for the current execution context" << std::endl;

            TrtCudaStream& stream = getStream(StreamType::kCOMPUTE);
            // Avoid capturing initialization calls by executing the enqueue function at least
            // once before starting CUDA graph capture.
            auto const ret = mEnqueue(stream);
            if (!ret)
            {
                throw std::runtime_error("Inference enqueue failed.");
            }
            stream.synchronize();

            mGraph.beginCapture(stream);
            // The built TRT engine may contain operations that are not permitted under CUDA graph capture mode.
            // When the stream is capturing, the enqueue call may return false if the current CUDA graph capture fails.
            if (mEnqueue(stream))
            {
                mGraph.endCapture(stream);
                mEnqueue = EnqueueFunction(EnqueueGraph(context, mGraph));
                sample::gLogInfo << "Successfully captured CUDA graph for the current execution context" << std::endl;
            }
            else
            {
                mGraph.endCaptureOnError(stream);
                // Ensure any CUDA error has been cleaned up.
                CHECK(cudaGetLastError());
                sample::gLogWarning << "The built TensorRT engine contains operations that are not permitted under "
                                       "CUDA graph capture mode."
                                    << std::endl;
                sample::gLogWarning << "The specified --useCudaGraph flag has been ignored. The inference will be "
                                       "launched without using CUDA graph launch."
                                    << std::endl;
            }
        }
    }

    Bindings& mBindings;

    TrtCudaGraph mGraph;
    EnqueueFunction mEnqueue;

    int32_t mStreamId{0};
    int32_t mNext{0};
    int32_t mDepth{2}; // default to double buffer to hide DMA transfers

    std::vector<bool> mActive;
    MultiStream mStream;
    std::vector<MultiEvent> mEvents;

    nvinfer1::IExecutionContext* mContext{nullptr};
};

bool inferenceLoop(std::vector<std::unique_ptr<Iteration>>& iStreams,bool skipTransfers)
{
    sample::gLogWarning << "--duration=-1 is specified, inference will run in an endless loop until"
                        << " aborted with CTRL-C (SIGINT)" << std::endl;
    while (true) 
    {
        for (auto& s : iStreams)
        {
            if (!s->query(skipTransfers))
            {
                return false;
            }
        }
        for (auto& s : iStreams)
        {
            s->sync(skipTransfers);
        }
    }
    
    return false;
}

void inferenceExecution(InferenceOptions const& inference, InferenceEnvironment& iEnv, SyncStruct& sync,
    int32_t const threadIdx, int32_t const streamsPerThread, int32_t device) noexcept
{
    try
    {
        CHECK(cudaSetDevice(device));

        std::vector<std::unique_ptr<Iteration>> iStreams;

        for (int32_t s = 0; s < streamsPerThread; ++s)
        {
            int32_t const streamId{threadIdx * streamsPerThread + s};
            sample::gLogInfo << "Thread: " << threadIdx << " Stream: " << streamId << std::endl;
            auto* iteration = new Iteration(streamId, inference, *iEnv.getContext(streamId), *iEnv.bindings[streamId]);
            if (inference.skipTransfers)
            {
                iteration->setInputData(true);
            }
            iStreams.emplace_back(iteration);
        }

        for (auto& s : iStreams)
        {
            s->wait(sync.gpuStart);
        }


        if (!inferenceLoop(iStreams,  inference.skipTransfers))
        {
            sync.mutex.lock();
            iEnv.error = true;
            sync.mutex.unlock();
        }

        if (inference.skipTransfers)
        {
            for (auto& s : iStreams)
            {
                s->fetchOutputData(true);
            }
        }

        sync.mutex.lock();
        sync.mutex.unlock();
    }
    catch (...)
    {
        sync.mutex.lock();
        iEnv.error = true;
        sync.mutex.unlock();
    }
}

inline std::thread makeThread(InferenceOptions const& inference, InferenceEnvironment& iEnv, SyncStruct& sync,
    int32_t threadIdx, int32_t streamsPerThread, int32_t device)
{
    return std::thread(inferenceExecution, std::cref(inference), std::ref(iEnv), std::ref(sync), threadIdx,
        streamsPerThread, device);
}

} // namespace

bool runInference(
    InferenceOptions const& inference, InferenceEnvironment& iEnv, int32_t device)
{
    SMP_RETVAL_IF_FALSE(!iEnv.safe, "Safe inference is not supported!", false, sample::gLogError);

    SyncStruct sync;

    sync.gpuStart.record(sync.mainStream);

    // When multiple streams are used, trtexec can run inference in two modes:
    // (1) if inference.threads is true, then run each stream on each thread.
    // (2) if inference.threads is false, then run all streams on the same thread.
    int32_t const numThreads = inference.threads ? inference.infStreams : 1;
    int32_t const streamsPerThread = inference.threads ? 1 : inference.infStreams;

    std::vector<std::thread> threads;
    for (int32_t threadIdx = 0; threadIdx < numThreads; ++threadIdx)
    {
        threads.emplace_back(makeThread(inference, iEnv, sync, threadIdx, streamsPerThread, device));
    }
    for (auto& th : threads)
    {
        th.join();
    }

    return !iEnv.error;
}


void Binding::fill(std::string const& fileName)
{
    loadFromFile(fileName, static_cast<char*>(buffer->getHostBuffer()), buffer->getSize());
}

void Binding::fill() {
    sample::gLogInfo << "Starting fill() with imageInfo.data: " << (imageInfo.data ? "present" : "null") << std::endl;
    
    if (imageInfo.data) {
        sample::gLogInfo << "Processing image of size: " << imageInfo.width << "x" << imageInfo.height << std::endl;
        size_t imageSize = imageInfo.width * imageInfo.height * 3;
        if (!pinnedBuffer) {
            CHECK(cudaMallocHost(&pinnedBuffer, imageSize));
        }
        memcpy(pinnedBuffer, imageInfo.data, imageSize);
        hasPendingImageProcess = true;
        sample::gLogInfo << "Image data copied to pinned memory" << std::endl;
    } else {
        sample::gLogInfo << "No image data, using random values" << std::endl;
        switch (dataType) {
        case nvinfer1::DataType::kFLOAT:
            fillBuffer<float>(buffer->getHostBuffer(), volume, -1.0F, 1.0F);
            break;
        }
    }
}


void Bindings::addBinding(TensorInfo const& tensorInfo, std::string const& fileName /*= ""*/)
{
    auto const b = tensorInfo.bindingIndex;
    while (mBindings.size() <= static_cast<size_t>(b))
    {
        mBindings.emplace_back();
        mDevicePointers.emplace_back();
    }
    mNames[tensorInfo.name] = b;
    mBindings[b].isInput = tensorInfo.isInput;
    mBindings[b].volume = tensorInfo.vol;
    mBindings[b].dataType = tensorInfo.dataType;
  

    if (mBindings[b].buffer == nullptr)
    {
        if (mUseManaged)
        {
            mBindings[b].buffer.reset(new UnifiedMirroredBuffer);
        }
        else
        {
            mBindings[b].buffer.reset(new DiscreteMirroredBuffer);
        }
    }
    // Some memory allocators return nullptr when allocating zero bytes, but TensorRT requires a non-null ptr
    // even for empty tensors, so allocate a dummy byte.
    if (tensorInfo.vol == 0)
    {
        mBindings[b].buffer->allocate(1);
    }
    else
    {
        mBindings[b].buffer->allocate(
            static_cast<size_t>(tensorInfo.vol) * static_cast<size_t>(dataTypeSize(tensorInfo.dataType)));
    }
    mDevicePointers[b] = mBindings[b].buffer->getDeviceBuffer();
    
    if (tensorInfo.isInput)
    {
        if (fileName.empty())
        {
            fill(b);
        }
        else
        {
            fill(b, fileName);
        }
    }
}

void** Bindings::getDeviceBuffers()
{
    return mDevicePointers.data();
}
void Bindings::transferInputToDevice(TrtCudaStream& stream) {

    sample::gLogInfo << "Starting transferInputToDevice" << std::endl;

    for (auto& b : mNames) {
        sample::gLogInfo << "Processing binding: isInput=" << mBindings[b.second].isInput 
                        << ", hasPendingImage=" << mBindings[b.second].hasPendingImageProcess << std::endl;
        
        if (mBindings[b.second].isInput) {
            if (mBindings[b.second].hasPendingImageProcess) {
                auto& binding = mBindings[b.second];
                sample::gLogInfo << "Image info: width=" << binding.imageInfo.width 
                               << ", height=" << binding.imageInfo.height 
                               << ", data=" << (binding.imageInfo.data != nullptr) 
                               << ", pinned=" << (binding.pinnedBuffer != nullptr) << std::endl;
                
                float* deviceDst = static_cast<float*>(binding.buffer->getDeviceBuffer());
                if (!deviceDst) {
                    sample::gLogError << "Device buffer is null!" << std::endl;
                    continue;
                }
                

                float scale = std::min(MODEL_INPUT_HEIGHT / (float)binding.imageInfo.height,
                                     MODEL_INPUT_WIDTH / (float)binding.imageInfo.width);
                
                sample::gLogInfo << "Scale computed: " << scale << std::endl;
                
                try {
                    float* d_warpMatrix;
                    float h_warpMatrix[6];
                    h_warpMatrix[0] = scale;
                    h_warpMatrix[1] = 0;
                    h_warpMatrix[2] = -scale * binding.imageInfo.width * 0.5 + MODEL_INPUT_WIDTH * 0.5;
                    h_warpMatrix[3] = 0;
                    h_warpMatrix[4] = scale;
                    h_warpMatrix[5] = -scale * binding.imageInfo.height * 0.5 + MODEL_INPUT_HEIGHT * 0.5;
                    cudaError_t error = cudaMalloc(&d_warpMatrix, sizeof(float) * 6);
                    if (error != cudaSuccess) {
                        sample::gLogError << "cudaMalloc失败: " << cudaGetErrorString(error) << std::endl;
                        return;
                    }
                    //CHECK(cudaMalloc(&d_warpMatrix, sizeof(float) * 6));
                    CHECK(cudaMemcpyAsync(d_warpMatrix, h_warpMatrix, sizeof(float) * 6, 
                                        cudaMemcpyHostToDevice, stream.get()));

                    int jobs = MODEL_INPUT_HEIGHT * MODEL_INPUT_WIDTH;
                    dim3 block(256, 1);
                    dim3 grid((jobs + block.x - 1) / block.x, 1);
                    
                    sample::gLogInfo << "Launching kernel with jobs=" << jobs << std::endl;
                    
                    warp_affine_bilinear_and_normalize_plane_kernel(
                        binding.pinnedBuffer,
                        binding.imageInfo.width * 3,
                        binding.imageInfo.width,
                        binding.imageInfo.height,
                        deviceDst,
                        MODEL_INPUT_WIDTH,
                        MODEL_INPUT_HEIGHT,
                        128,
                        d_warpMatrix,
                        binding.imageInfo.needNormalize,
                        jobs
                    );
                    
                    //CHECK(cudaGetLastError());
                    CHECK(cudaFreeAsync(d_warpMatrix, stream.get()));
                    sample::gLogInfo << "Kernel completed successfully" << std::endl;
                    
                    binding.hasPendingImageProcess = false;
                }
                catch (const std::exception& e) {
                    sample::gLogError << "Error during image processing: " << e.what() << std::endl;
                }
            } else {
                sample::gLogInfo << "No pending image processing, using regular host to device transfer" << std::endl;
                mBindings[b.second].buffer->hostToDevice(stream);
            }
        }
    }
}
void Bindings::preprocessImage(uint8_t* imgData, int width, int height, bool normalize, cudaStream_t stream) 
{
    sample::gLogInfo << "Starting preprocessImage with dimensions " << width << "x" << height << std::endl;
    
    imageInfo.data = imgData;
    imageInfo.width = width;
    imageInfo.height = height;
    imageInfo.needNormalize = normalize;

    auto inputBindings = getInputBindings();
    sample::gLogInfo << "Found " << inputBindings.size() << " inputdsfsdfs4544bindings" << std::endl;

    for (const auto& binding : inputBindings) {
        sample::gLogInfo << "Processing binding " << binding.first << " with index " << binding.second << std::endl;
        
        int bindingIdx = binding.second;
        if (bindingIdx >= mBindings.size() || !mBindings[bindingIdx].isInput) {
            sample::gLogInfo << "Skipping invalid binding" << std::endl;
            continue;
        }

        mBindings[bindingIdx].setupImagePreprocess(imgData, width, height, normalize);
        fill(bindingIdx);
    }
    
    sample::gLogInfo << "Image preprocessing setup complete" << std::endl;
}
void Bindings::transferOutputToHost(TrtCudaStream& stream)
{
    for (auto& b : mNames)
    {
        if (!mBindings[b.second].isInput)
        {
            if (mBindings[b.second].outputAllocator != nullptr)
            {
                mBindings[b.second].outputAllocator->getBuffer()->deviceToHost(stream);
            }
            else
            {
                mBindings[b.second].buffer->deviceToHost(stream);
            }
        }
    }
}

void Bindings::dumpBindingValues(nvinfer1::IExecutionContext const& context, int32_t binding, std::ostream& os,
    std::string const& separator /*= " "*/, int32_t batch /*= 1*/) const
{
    auto const tensorName = context.getEngine().getIOTensorName(binding);
    Dims dims = context.getTensorShape(tensorName);
    Dims strides = context.getTensorStrides(tensorName);
    int32_t vectorDim = context.getEngine().getTensorVectorizedDim(tensorName);
    int32_t const spv = context.getEngine().getTensorComponentsPerElement(tensorName);

    mBindings[binding].dump(os, dims, strides, vectorDim, spv, separator);
}

namespace
{

std::string genFilenameSafeString(std::string const& s)
{
    std::string res = s;
    static std::string const allowedSpecialChars{"._-,"};
    for (auto& c : res)
    {
        if (!isalnum(c) && allowedSpecialChars.find(c) == std::string::npos)
        {
            c = '_';
        }
    }
    return res;
}

Dims getBindingDimensions(nvinfer1::IExecutionContext const& context, std::string const& name)
{
    return context.getTensorShape(name.c_str());
}
} // namespace

void Bindings::dumpBindingDimensions(
    std::string const& name, nvinfer1::IExecutionContext const& context, std::ostream& os) const
{
    auto const dims = context.getTensorShape(name.c_str());
    // Do not add a newline terminator, because the caller may be outputting a JSON string.
    os << dims;
}

std::unordered_map<std::string, int> Bindings::getBindings(std::function<bool(Binding const&)> predicate) const
{
    std::unordered_map<std::string, int> bindings;
    for (auto const& n : mNames)
    {
        auto const binding = n.second;
        if (predicate(mBindings[binding]))
        {
            bindings.insert(n);
        }
    }
    return bindings;
}

bool Bindings::setTensorAddresses(nvinfer1::IExecutionContext& context) const
{
    for (auto const& b : mNames)
    {
        auto const name = b.first.c_str();
        auto const location = context.getEngine().getTensorLocation(name);
        if (location == TensorLocation::kDEVICE)
        {
            if (mBindings[b.second].outputAllocator != nullptr)
            {
                if (!context.setOutputAllocator(name, mBindings[b.second].outputAllocator.get()))
                {
                    return false;
                }
            }
            else
            {
                if (!context.setTensorAddress(name, mDevicePointers[b.second]))
                {
                    return false;
                }
            }
        }
    }
    return true;
}

} // namespace sample
