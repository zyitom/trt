#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <opencv2/highgui.hpp>
#include <sstream>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "opencv2/videoio.hpp"
#include "sampleOptions.h"
#include "logger.h"
#include "sampleInference.h"
#include "sampleDevice.h"
#include "sampleEngines.h"

using namespace nvinfer1;
using namespace sample;


template<typename T>
class ThreadSafeQueue {
public:
    ThreadSafeQueue(size_t maxSize = 4 ) : maxSize(maxSize) {}
    
    void push(T item) {
        std::lock_guard<std::mutex> lock(mutex);

        while (queue.size() >= maxSize) {
            queue.pop();
        }
        queue.push(std::move(item));
        cond.notify_one();
    }
    
    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex);
        cond.wait(lock, [this]{ return !queue.empty() || stop; });
        
        if (queue.empty() && stop) {
            return false;
        }
        
        item = std::move(queue.front());
        queue.pop();
        return true;
    }
    
    void setStop() {
        std::lock_guard<std::mutex> lock(mutex);
        stop = true;
        cond.notify_all();
    }
    
    bool empty() {
        std::lock_guard<std::mutex> lock(mutex);
        return queue.empty();
    }

private:
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cond;
    bool stop = false;
    const size_t maxSize;
};


struct FrameData {
    cv::Mat frame;
    int frameCount;
};

namespace {
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


void captureThread(cv::VideoCapture& cap, ThreadSafeQueue<FrameData>& frameQueue, 
                  std::atomic<bool>& running, std::unique_ptr<InferenceEnvironment>& iEnv) {
    int frameCount = 0;
    auto start = std::chrono::steady_clock::now();
    
    while (running) {
        cv::Mat frame;  
        if (!cap.read(frame)) {
            sample::gLogInfo << "Video ended or failed to read frame" << std::endl;
            break;
        }
        
        if (frame.empty()) {
            sample::gLogError << "Empty frame encountered" << std::endl;
            continue;
        }

        FrameData frameData;
        frameData.frameCount = ++frameCount;
        frameData.frame = frame.clone(); 
        frame.release(); 
        
        auto inputBindings = iEnv->bindings[0]->getInputBindings();
        if (inputBindings.empty()) {
            sample::gLogError << "No input bindings found!" << std::endl;
            break;
        }
        
     
        for (const auto& [name, bindingIdx] : inputBindings) {
            sample::gLogInfo << "Processing binding " << name << " for frame " << frameCount << std::endl;
            iEnv->bindings[0]->preprocessImage(
                frameData.frame.data,
                frameData.frame.cols,
                frameData.frame.rows,
                true,
                nullptr
            );
        }

        frameQueue.push(std::move(frameData));  
        
       
        if (frameCount % 100 == 0) {
            auto now = std::chrono::steady_clock::now();
            float fps = frameCount * 1000.0f / 
                std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
            sample::gLogInfo << "Captured and preprocessed " << frameCount << " frames at " << fps << " FPS" << std::endl;
        }
        
      
        
    }
    
    frameQueue.setStop();
}

int main(int argc, char** argv) {
    try {
        Arguments args;
        AllOptions options;
        
        // 设置选项
        options.build.load = true;
        options.build.engine = "/home/zyi/model.engine";
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
        
        if (!setUpInference(*iEnv, options.inference, options.system)) {
            sample::gLogError << "Inference set up failed" << std::endl;
            return -1;
        }
        
    
        cv::VideoCapture cap("/home/zyi/Videos/2024-05-30_19_47_S.avi");
        if (!cap.isOpened()) {
            sample::gLogError << "Failed to open video file" << std::endl;
            return -1;
        }
        

        ThreadSafeQueue<FrameData> frameQueue(3);  
        std::atomic<bool> running(true);

    
        std::thread captureThreadObj(captureThread, 
                                   std::ref(cap), 
                                   std::ref(frameQueue), 
                                   std::ref(running), 
                                   std::ref(iEnv));
        
   
        while (running) {
            FrameData frameData;
            if (!frameQueue.pop(frameData)) {
                break;
            }
            
      
            if (!runInference(options.inference, *iEnv, options.system.device)) {
                sample::gLogError << "Inference failed for frame " << frameData.frameCount << std::endl;
            }
            
         
            frameData.frame.release();
            
            

        }
        
      
        running = false;
        if (captureThreadObj.joinable()) {
            captureThreadObj.join();
        }
        
        return 0;
    }
    catch (std::exception const& e) {
        sample::gLogError << "Error: " << e.what() << std::endl;
        return -1;
    }
}