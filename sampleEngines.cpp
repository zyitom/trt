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
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "NvInfer.h"


#include "ErrorRecorder.h"
#include "common.h"
#include "logger.h"
#include "sampleDevice.h"
#include "sampleEngines.h"
#include "sampleOptions.h"
#include "sampleUtils.h"

using namespace nvinfer1;

namespace sample
{

    std::function<void*(void*, int32_t)> pCreateInferRuntimeInternal{};

    bool initNvinfer() {
        pCreateInferRuntimeInternal = createInferRuntime_INTERNAL;
        return true;
    }

IRuntime* createRuntime() {
    if (!initNvinfer()) {
        return {};
    }
    return static_cast<IRuntime*>(pCreateInferRuntimeInternal(&gLogger.getTRTLogger(), NV_TENSORRT_VERSION));
}
nvinfer1::ICudaEngine* LazilyDeserializedEngine::get()
{
    SMP_RETVAL_IF_FALSE(
        !mIsSafe, "Safe mode is enabled, but trying to get standard engine!", nullptr, sample::gLogError);

    if (mEngine == nullptr)
    {

        SMP_RETVAL_IF_FALSE(getAsyncFileReader().isOpen() || getFileReader().isOpen() || !getBlob().empty(),
            "Engine is empty. Nothing to deserialize!", nullptr, sample::gLogError);
        using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;
        using duration = std::chrono::duration<float>;
        time_point const deserializeStartTime{std::chrono::high_resolution_clock::now()};

        if (mLeanDLLPath.empty())
        {
            mRuntime.reset(createRuntime());
        }
        else
        {
            mParentRuntime.reset(createRuntime());
            ASSERT(mParentRuntime.get() != nullptr);

            mRuntime.reset(mParentRuntime->loadRuntime(mLeanDLLPath.c_str()));
        }
        ASSERT(mRuntime.get() != nullptr);

        if (mVersionCompatible)
        {
            // Application needs to opt into allowing deserialization of engines with embedded lean runtime.
            mRuntime->setEngineHostCodeAllowed(true);
        }

        if (!mTempdir.empty())
        {
            mRuntime->setTemporaryDirectory(mTempdir.c_str());
        }

        mRuntime->setTempfileControlFlags(mTempfileControls);

        SMP_RETVAL_IF_FALSE(mRuntime != nullptr, "runtime creation failed", nullptr, sample::gLogError);
        if (mDLACore != -1)
        {
            mRuntime->setDLACore(mDLACore);
        }
        mRuntime->setErrorRecorder(&gRecorder);

        for (auto const& pluginPath : mDynamicPlugins)
        {
            mRuntime->getPluginRegistry().loadLibrary(pluginPath.c_str());
        }

        if (getFileReader().isOpen())
        {
            mEngine.reset(mRuntime->deserializeCudaEngine(getFileReader()));
        }
        else if (getAsyncFileReader().isOpen())
        {
            mEngine.reset(mRuntime->deserializeCudaEngine(getAsyncFileReader()));
        }
        else
        {
            auto const& engineBlob = getBlob();
            mEngine.reset(mRuntime->deserializeCudaEngine(engineBlob.data, engineBlob.size));
        }
        SMP_RETVAL_IF_FALSE(mEngine != nullptr, "Engine deserialization failed", nullptr, sample::gLogError);

        time_point const deserializeEndTime{std::chrono::high_resolution_clock::now()};
        sample::gLogInfo << "Engine deserialized in " << duration(deserializeEndTime - deserializeStartTime).count()
                         << " sec." << std::endl;
    }

    return mEngine.get();
}

nvinfer1::ICudaEngine* LazilyDeserializedEngine::release()
{
    return mEngine.release();
}

bool loadStreamingEngineToBuildEnv(std::string const& filepath, BuildEnvironment& env, std::ostream& err)
{
    auto& reader = env.engine.getFileReader();
    SMP_RETVAL_IF_FALSE(reader.open(filepath), "", false, err << "Error opening engine file: " << filepath);
    return true;
}

bool loadAsyncStreamingEngineToBuildEnv(std::string const& filepath, BuildEnvironment& env, std::ostream& err)
{
    auto& asyncReader = env.engine.getAsyncFileReader();
    SMP_RETVAL_IF_FALSE(asyncReader.open(filepath), "", false, err << "Error opening engine file: " << filepath);
    return true;
}



bool getEngineBuildEnv(
    const ModelOptions& model, BuildOptions const& build, SystemOptions& sys, BuildEnvironment& env, std::ostream& err)
{
    bool createEngineSuccess{false};


        if (build.safe)
        {
            createEngineSuccess = loadEngineToBuildEnv(build.engine, env, err);
        }
      

    SMP_RETVAL_IF_FALSE(createEngineSuccess, "Failed to create engine from model or file.", false, err);

    return true;
}

bool loadEngineToBuildEnv(std::string const& filepath, BuildEnvironment& env, std::ostream& err)
{
    auto const tBegin = std::chrono::high_resolution_clock::now();
    std::ifstream engineFile(filepath, std::ios::binary);
    SMP_RETVAL_IF_FALSE(engineFile.good(), "", false, err << "Error opening engine file: " << filepath);
    engineFile.seekg(0, std::ifstream::end);
    int64_t fsize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);

    std::vector<uint8_t> engineBlob(fsize);
    engineFile.read(reinterpret_cast<char*>(engineBlob.data()), fsize);
    SMP_RETVAL_IF_FALSE(engineFile.good(), "", false, err << "Error loading engine file: " << filepath);
    auto const tEnd = std::chrono::high_resolution_clock::now();
    float const loadTime = std::chrono::duration<float>(tEnd - tBegin).count();
    sample::gLogInfo << "Engine loaded in " << loadTime << " sec." << std::endl;
    sample::gLogInfo << "Loaded engine with size: " << (fsize / 1.0_MiB) << " MiB" << std::endl;

    env.engine.setBlob(std::move(engineBlob));

    return true;
}

}