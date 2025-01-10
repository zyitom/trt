#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <vector>

#include "NvInfer.h"

#include "NvInferPlugin.h"
#include "sampleOptions.h"
#include "logger.h"
#include "sampleInference.h"
#include "sampleDevice.h"
#include "sampleEngines.h"

using namespace nvinfer1;
using namespace sample;
namespace{
std::function<void*(void*, int32_t)> pCreateInferRuntimeInternal{};

bool initNvinfer() {
    pCreateInferRuntimeInternal = createInferRuntime_INTERNAL;
    return true;
}

}
IRuntime* createRuntime() {
    if (!initNvinfer()) {
        return {};
    }
    return static_cast<IRuntime*>(pCreateInferRuntimeInternal(&gLogger.getTRTLogger(), NV_TENSORRT_VERSION));
}
int main(int argc, char** argv)
{
    try
    {
        Arguments args;
        AllOptions options;
        

        options.build.load = true; 
        options.build.engine = "/home/zyi/Desktop/fl68-bigpa-9700.engine";  
        options.system.device = 0;  
        options.inference.batch = 1;  
        options.inference.iterations = -1; 
        options.inference.infStreams = 4;
        options.inference.threads = true;
        options.inference.graph = true;
        setCudaDevice(options.system.device, sample::gLogInfo);

        std::unique_ptr<BuildEnvironment> bEnv(new BuildEnvironment(
            options.build.safe,
            options.build.versionCompatible,
            options.system.DLACore,
            options.build.tempdir,
            options.build.tempfileControls,
            options.build.leanDLLPath
        ));

        if (!loadEngineToBuildEnv(options.build.engine, *bEnv, sample::gLogError)) {
            sample::gLogError << "Failed to load engine" << std::endl;
            return -1;
        }

        std::unique_ptr<InferenceEnvironment> iEnv(new InferenceEnvironment(*bEnv));

        if (!setUpInference(*iEnv, options.inference, options.system))
        {
            sample::gLogError << "Inference set up failed" << std::endl;
            return -1;
        }

        sample::gLogInfo << "Starting inference" << std::endl;
        
        if (!runInference(options.inference, *iEnv, options.system.device))
        {
            sample::gLogError << "Error occurred during inference" << std::endl;
            return -1;
        }

        return 0;
    }
    catch (std::exception const& e)
    {
        sample::gLogError << "Error: " << e.what() << std::endl;
        return -1;
    }
}