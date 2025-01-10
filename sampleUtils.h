#ifndef SAMPLE_UTILS_H
#define SAMPLE_UTILS_H

#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda.h>
#include <cuda_fp16.h>

#include "NvInfer.h"

#include "common.h"
#include "logger.h"



#define SMP_RETVAL_IF_FALSE(condition, msg, retval, err)                                                               \
{                                                                                                                  \
if ((condition) == false)                                                                                      \
{                                                                                                              \
(err) << (msg) << std::endl;                                                                               \
return retval;                                                                                             \
}                                                                                                              \
}

namespace sample {
    size_t dataTypeSize(nvinfer1::DataType dataType);

    template <typename T>
    inline T roundUp(T m, T n)
    {
        return ((m + n - 1) / n) * n;
    }



    int64_t volume(nvinfer1::Dims const& dims, nvinfer1::Dims const& strides, int32_t vecDim, int32_t comps, int32_t batch);

    using samplesCommon::volume;


    void loadFromFile(std::string const& fileName, char* dst, size_t size);

    nvinfer1::Dims toDims(std::vector<int32_t> const& vec);

    template <typename T>
    void dumpBuffer(void const* buffer, std::string const& separator, std::ostream& os, nvinfer1::Dims const& dims,
        nvinfer1::Dims const& strides, int32_t vectorDim, int32_t spv);

    bool matchStringWithOneWildcard(std::string const& pattern, std::string const& target);


    template <typename T>
    typename std::unordered_map<std::string, T>::const_iterator findPlausible(
        std::unordered_map<std::string, T> const& map, std::string const& target)
    {
        auto res = map.find(target);
        if (res == map.end())
        {
            res = std::find_if(
                map.begin(), map.end(), [&](typename std::unordered_map<std::string, T>::value_type const& item) {
                    return matchStringWithOneWildcard(item.first, target);
                });
        }
        return res;
    }
    template <typename T, typename std::enable_if_t<std::is_integral_v<T>, bool> = true>
    void fillBuffer(void* buffer, int64_t volume, T min, T max);

    template <typename T, typename std::enable_if_t<!std::is_integral_v<T>, int32_t> = 0>
    void fillBuffer(void* buffer, int64_t volume, T min, T max);
    std::vector<std::string> splitToStringVec(std::string const& option, char separator, int64_t maxSplit = -1);

   

}
#endif // SAMPLE_UTILS_H