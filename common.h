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

#ifndef TENSORRT_COMMON_H
#define TENSORRT_COMMON_H
#include "NvInfer.h"
#if !TRT_WINML
#include "NvInferPlugin.h"
#endif
#include "logger.h"
#include "safeCommon.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda_runtime_api.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <new>
#include <numeric>
#include <ratio>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef _MSC_VER
// For loadLibrary
// Needed so that the max/min definitions in windows.h do not conflict with std::max/min.
#define NOMINMAX
#include <windows.h>
#undef NOMINMAX
#else
#include <dlfcn.h>
#endif

#ifdef _MSC_VER
#define FN_NAME __FUNCTION__
#else
#define FN_NAME __func__
#endif

#if defined(__aarch64__) || defined(__QNX__)
#define ENABLE_DLA_API 1
#endif

#define CHECK_RETURN_W_MSG(status, val, errMsg)                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(status))                                                                                                 \
        {                                                                                                              \
            sample::gLogError << errMsg << " Error in " << __FILE__ << ", function " << FN_NAME << "(), line " << __LINE__     \
                      << std::endl;                                                                                    \
            return val;                                                                                                \
        }                                                                                                              \
    } while (0)

#undef ASSERT
#define ASSERT(condition)                                                   \
    do                                                                      \
    {                                                                       \
        if (!(condition))                                                   \
        {                                                                   \
            sample::gLogError << "Assertion failure: " << #condition << std::endl;  \
            exit(EXIT_FAILURE);                                                       \
        }                                                                   \
    } while (0)


#define CHECK_RETURN(status, val) CHECK_RETURN_W_MSG(status, val, "")

#define OBJ_GUARD(A) std::unique_ptr<A, void (*)(A * t)>

template <typename T, typename T_>
OBJ_GUARD(T)
makeObjGuard(T_* t)
{
    static_assert(std::is_base_of_v<T, T_> || std::is_same_v<T, T_>);
    auto deleter = [](T* t) { delete t; };
    return std::unique_ptr<T, decltype(deleter)>{static_cast<T*>(t), deleter};
}

constexpr long double operator"" _GiB(long double val)
{
    return val * (1 << 30);
}
constexpr long double operator"" _MiB(long double val)
{
    return val * (1 << 20);
}
constexpr long double operator"" _KiB(long double val)
{
    return val * (1 << 10);
}


extern nvinfer1::IBuilder* createBuilder();

namespace samplesCommon
{

// Swaps endianness of an integral type.
template <typename T, typename std::enable_if_t<std::is_integral_v<T>, int> = 0>
inline T swapEndianness(const T& value)
{
    uint8_t bytes[sizeof(T)];
    for (int i = 0; i < static_cast<int>(sizeof(T)); ++i)
    {
        bytes[sizeof(T) - 1 - i] = *(reinterpret_cast<const uint8_t*>(&value) + i);
    }
    return *reinterpret_cast<T*>(bytes);
}

class HostMemory
{
public:
    HostMemory() = delete;
    virtual void* data() const noexcept
    {
        return mData;
    }
    virtual std::size_t size() const noexcept
    {
        return mSize;
    }
    virtual nvinfer1::DataType type() const noexcept
    {
        return mType;
    }
    virtual ~HostMemory() {}

protected:
    HostMemory(std::size_t size, nvinfer1::DataType type)
        : mData{nullptr}
        , mSize(size)
        , mType(type)
    {
    }
    void* mData;
    std::size_t mSize;
    nvinfer1::DataType mType;
};

template <typename ElemType, nvinfer1::DataType dataType>
class TypedHostMemory : public HostMemory
{
public:
    explicit TypedHostMemory(std::size_t size)
        : HostMemory(size, dataType)
    {
        mData = new ElemType[size];
    };
    ~TypedHostMemory() noexcept override
    {
        delete[](ElemType*) mData;
    }
    ElemType* raw() noexcept
    {
        return static_cast<ElemType*>(data());
    }
};

using FloatMemory = TypedHostMemory<float, nvinfer1::DataType::kFLOAT>;
using HalfMemory = TypedHostMemory<uint16_t, nvinfer1::DataType::kHALF>;
using ByteMemory = TypedHostMemory<uint8_t, nvinfer1::DataType::kINT8>;

inline void* safeCudaMalloc(size_t memSize)
{
    void* deviceMem;
    CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr)
    {
        std::cerr << "Out of memory" << std::endl;
        exit(EXIT_FAILURE);
    }
    return deviceMem;
}

inline bool isDebug()
{
    return (std::getenv("TENSORRT_DEBUG") ? true : false);
}

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        delete obj;
    }
};

template <typename T>
using SampleUniquePtr = std::unique_ptr<T>;

static auto StreamDeleter = [](cudaStream_t* pStream) {
    if (pStream)
    {
        static_cast<void>(cudaStreamDestroy(*pStream));
        delete pStream;
    }
};

inline std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> makeCudaStream()
{
    std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> pStream(new cudaStream_t, StreamDeleter);
    if (cudaStreamCreateWithFlags(pStream.get(), cudaStreamNonBlocking) != cudaSuccess)
    {
        pStream.reset(nullptr);
    }

    return pStream;
}

//! Return vector of indices that puts magnitudes of sequence in descending order.
template <class Iter>
std::vector<size_t> argMagnitudeSort(Iter begin, Iter end)
{
    std::vector<size_t> indices(end - begin);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&begin](size_t i, size_t j) { return std::abs(begin[j]) < std::abs(begin[i]); });
    return indices;
}

inline bool readReferenceFile(const std::string& fileName, std::vector<std::string>& refVector)
{
    std::ifstream infile(fileName);
    if (!infile.is_open())
    {
        std::cout << "ERROR: readReferenceFile: Attempting to read from a file that is not open." << std::endl;
        return false;
    }
    std::string line;
    while (std::getline(infile, line))
    {
        if (line.empty())
            continue;
        refVector.push_back(line);
    }
    infile.close();
    return true;
}

template <typename T>
std::vector<std::string> classify(
    const std::vector<std::string>& refVector, const std::vector<T>& output, const size_t topK)
{
    const auto inds = samplesCommon::argMagnitudeSort(output.cbegin(), output.cend());
    std::vector<std::string> result;
    result.reserve(topK);
    for (size_t k = 0; k < topK; ++k)
    {
        result.push_back(refVector[inds[k]]);
    }
    return result;
}

// Returns indices of highest K magnitudes in v.
template <typename T>
std::vector<size_t> topKMagnitudes(const std::vector<T>& v, const size_t k)
{
    std::vector<size_t> indices = samplesCommon::argMagnitudeSort(v.cbegin(), v.cend());
    indices.resize(k);
    return indices;
}

template <typename T>
bool readASCIIFile(const std::string& fileName, const size_t size, std::vector<T>& out)
{
    std::ifstream infile(fileName);
    if (!infile.is_open())
    {
        std::cout << "ERROR readASCIIFile: Attempting to read from a file that is not open." << std::endl;
        return false;
    }
    out.clear();
    out.reserve(size);
    out.assign(std::istream_iterator<T>(infile), std::istream_iterator<T>());
    infile.close();
    return true;
}

template <typename T>
bool writeASCIIFile(const std::string& fileName, const std::vector<T>& in)
{
    std::ofstream outfile(fileName);
    if (!outfile.is_open())
    {
        std::cout << "ERROR: writeASCIIFile: Attempting to write to a file that is not open." << std::endl;
        return false;
    }
    for (auto fn : in)
    {
        outfile << fn << "\n";
    }
    outfile.close();
    return true;
}

inline void print_version()
{
    std::cout << "  TensorRT version: " << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH
              << "." << NV_TENSORRT_BUILD << std::endl;
}

inline std::string getFileType(const std::string& filepath)
{
    return filepath.substr(filepath.find_last_of(".") + 1);
}

inline std::string toLower(const std::string& inp)
{
    std::string out = inp;
    std::transform(out.begin(), out.end(), out.begin(), ::tolower);
    return out;
}

inline float getMaxValue(const float* buffer, int64_t size)
{
    assert(buffer != nullptr);
    assert(size > 0);
    return *std::max_element(buffer, buffer + size);
}

// Ensures that every tensor used by a network has a dynamic range set.
//
// All tensors in a network must have a dynamic range specified if a calibrator is not used.
// This function is just a utility to globally fill in missing scales and zero-points for the entire network.
//
// If a tensor does not have a dyanamic range set, it is assigned inRange or outRange as follows:
//
// * If the tensor is the input to a layer or output of a pooling node, its dynamic range is derived from inRange.
// * Otherwise its dynamic range is derived from outRange.
//
// The default parameter values are intended to demonstrate, for final layers in the network,
// cases where dynamic ranges are asymmetric.
//
// The default parameter values choosen arbitrarily. Range values should be choosen such that
// we avoid underflow or overflow. Also range value should be non zero to avoid uniform zero scale tensor.
inline void setAllDynamicRanges(nvinfer1::INetworkDefinition* network, float inRange = 2.0F, float outRange = 4.0F)
{
    // Ensure that all layer inputs have a scale.
    for (int i = 0; i < network->getNbLayers(); i++)
    {
        auto layer = network->getLayer(i);
        for (int j = 0; j < layer->getNbInputs(); j++)
        {
            nvinfer1::ITensor* input{layer->getInput(j)};
            // Optional inputs are nullptr here and are from RNN layers.
            if (input != nullptr && !input->dynamicRangeIsSet())
            {
                ASSERT(input->setDynamicRange(-inRange, inRange));
            }
        }
    }

    // Ensure that all layer outputs have a scale.
    // Tensors that are also inputs to layers are ingored here
    // since the previous loop nest assigned scales to them.
    for (int i = 0; i < network->getNbLayers(); i++)
    {
        auto layer = network->getLayer(i);
        for (int j = 0; j < layer->getNbOutputs(); j++)
        {
            nvinfer1::ITensor* output{layer->getOutput(j)};
            // Optional outputs are nullptr here and are from RNN layers.
            if (output != nullptr && !output->dynamicRangeIsSet())
            {
                // Pooling must have the same input and output scales.
                if (layer->getType() == nvinfer1::LayerType::kPOOLING)
                {
                    ASSERT(output->setDynamicRange(-inRange, inRange));
                }
                else
                {
                    ASSERT(output->setDynamicRange(-outRange, outRange));
                }
            }
        }
    }
}

inline void setDummyInt8DynamicRanges(const nvinfer1::IBuilderConfig* c, nvinfer1::INetworkDefinition* n)
{
    // Set dummy per-tensor dynamic range if Int8 mode is requested.
    if (c->getFlag(nvinfer1::BuilderFlag::kINT8))
    {
        sample::gLogWarning << "Int8 calibrator not provided. Generating dummy per-tensor dynamic range. Int8 accuracy "
                               "is not guaranteed."
                            << std::endl;
        setAllDynamicRanges(n);
    }
}

inline void enableDLA(
    nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, int useDLACore, bool allowGPUFallback = true)
{
    if (useDLACore >= 0)
    {
        if (builder->getNbDLACores() == 0)
        {
            std::cerr << "Trying to use DLA core " << useDLACore << " on a platform that doesn't have any DLA cores"
                      << std::endl;
            assert("Error: use DLA core on a platfrom that doesn't have any DLA cores" && false);
        }
        if (allowGPUFallback)
        {
            config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
        }
        if (!config->getFlag(nvinfer1::BuilderFlag::kINT8))
        {
            // User has not requested INT8 Mode.
            // By default run in FP16 mode. FP32 mode is not permitted.
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
        config->setDLACore(useDLACore);
    }
}

inline int32_t parseDLA(int32_t argc, char** argv)
{
    for (int32_t i = 1; i < argc; i++)
    {
        if (strncmp(argv[i], "--useDLACore=", 13) == 0)
        {
            return std::stoi(argv[i] + 13);
        }
    }
    return -1;
}

inline uint32_t getElementSize(nvinfer1::DataType t) noexcept
{
    switch (t)
    {
    case nvinfer1::DataType::kINT64: return 8;
    case nvinfer1::DataType::kINT32:
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kBF16:
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kUINT8:
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kFP8: return 1;
    case nvinfer1::DataType::kINT4:
        ASSERT(false && "Element size is not implemented for sub-byte data-types");
    }
    return 0;
}

inline int64_t volume(nvinfer1::Dims const& dims, int32_t start, int32_t stop)
{
    ASSERT(start >= 0);
    ASSERT(start <= stop);
    ASSERT(stop <= dims.nbDims);
    ASSERT(std::all_of(dims.d + start, dims.d + stop, [](int32_t x) { return x >= 0; }));
    return std::accumulate(dims.d + start, dims.d + stop, int64_t{1}, std::multiplies<int64_t>{});
}



struct BBox
{
    float x1, y1, x2, y2;
};



inline std::vector<std::string> splitString(std::string str, char delimiter = ',')
{
    std::vector<std::string> splitVect;
    std::stringstream ss(str);
    std::string substr;

    while (ss.good())
    {
        getline(ss, substr, delimiter);
        splitVect.emplace_back(std::move(substr));
    }
    return splitVect;
}

inline int getC(nvinfer1::Dims const& d)
{
    return d.nbDims >= 3 ? d.d[d.nbDims - 3] : 1;
}

inline int getH(const nvinfer1::Dims& d)
{
    return d.nbDims >= 2 ? d.d[d.nbDims - 2] : 1;
}

inline int getW(const nvinfer1::Dims& d)
{
    return d.nbDims >= 1 ? d.d[d.nbDims - 1] : 1;
}

//! Platform-agnostic wrapper around dynamic libraries.
class DynamicLibrary
{
public:
    explicit DynamicLibrary(std::string const& name)
        : mLibName{name}
    {
#if defined(_WIN32)
        mHandle = LoadLibraryA(name.c_str());
#else // defined(_WIN32)
        int32_t flags{RTLD_LAZY};
#if ENABLE_ASAN
        // https://github.com/google/sanitizers/issues/89
        // asan doesn't handle module unloading correctly and there are no plans on doing
        // so. In order to get proper stack traces, don't delete the shared library on
        // close so that asan can resolve the symbols correctly.
        flags |= RTLD_NODELETE;
#endif // ENABLE_ASAN

        mHandle = dlopen(name.c_str(), flags);
#endif // defined(_WIN32)

        if (mHandle == nullptr)
        {
            std::string errorStr{};
#if !defined(_WIN32)
            errorStr = std::string{" due to "} + std::string{dlerror()};
#endif
            throw std::runtime_error("Unable to open library: " + name + errorStr);
        }
    }

    DynamicLibrary(DynamicLibrary const&) = delete;
    DynamicLibrary(DynamicLibrary const&&) = delete;

    //!
    //! Retrieve a function symbol from the loaded library.
    //!
    //! \return the loaded symbol on success
    //! \throw std::invalid_argument if loading the symbol failed.
    //!
    template <typename Signature>
    std::function<Signature> symbolAddress(char const* name)
    {
        if (mHandle == nullptr)
        {
            throw std::runtime_error("Handle to library is nullptr.");
        }
        void* ret;
#if defined(_MSC_VER)
        ret = static_cast<void*>(GetProcAddress(static_cast<HMODULE>(mHandle), name));
#else
        ret = dlsym(mHandle, name);
#endif
        if (ret == nullptr)
        {
            std::string const kERROR_MSG(mLibName + ": error loading symbol: " + std::string(name));
            throw std::invalid_argument(kERROR_MSG);
        }
        return reinterpret_cast<Signature*>(ret);
    }

    ~DynamicLibrary()
    {
        try
        {
#if defined(_WIN32)
            ASSERT(static_cast<bool>(FreeLibrary(static_cast<HMODULE>(mHandle))));
#else
            ASSERT(dlclose(mHandle) == 0);
#endif
        }
        catch (...)
        {
            sample::gLogError << "Unable to close library: " << mLibName << std::endl;
        }
    }

private:
    std::string mLibName{}; //!< Name of the DynamicLibrary
    void* mHandle{};        //!< Handle to the DynamicLibrary
};

inline std::unique_ptr<DynamicLibrary> loadLibrary(std::string const& path)
{
    // make_unique not available until C++14 - we still need to support C++11 builds.
    return std::unique_ptr<DynamicLibrary>(new DynamicLibrary{path});
}

inline int32_t getMaxPersistentCacheSize()
{
    int32_t deviceIndex{};
    CHECK(cudaGetDevice(&deviceIndex));

    int32_t maxPersistentL2CacheSize{};
#if CUDART_VERSION >= 11030 && !TRT_WINML
    CHECK(cudaDeviceGetAttribute(&maxPersistentL2CacheSize, cudaDevAttrMaxPersistingL2CacheSize, deviceIndex));
#endif

    return maxPersistentL2CacheSize;
}

inline bool isDataTypeSupported(nvinfer1::DataType dataType)
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(createBuilder());
    if (!builder)
    {
        return false;
    }

    return true;
}
} // namespace samplesCommon

inline std::ostream& operator<<(std::ostream& os, const nvinfer1::Dims& dims)
{
    os << "(";
    for (int i = 0; i < dims.nbDims; ++i)
    {
        os << (i ? ", " : "") << dims.d[i];
    }
    return os << ")";
}

#endif // TENSORRT_COMMON_H
