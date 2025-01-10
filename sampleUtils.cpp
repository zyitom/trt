#include "sampleUtils.h"

namespace sample {
    size_t dataTypeSize(nvinfer1::DataType dataType)
    {
        switch (dataType)
        {
            case nvinfer1::DataType::kINT64: return 8U;
            case nvinfer1::DataType::kINT32:
            case nvinfer1::DataType::kFLOAT: return 4U;
            case nvinfer1::DataType::kBF16:
            case nvinfer1::DataType::kHALF: return 2U;
            case nvinfer1::DataType::kBOOL:
            case nvinfer1::DataType::kUINT8:
            case nvinfer1::DataType::kINT8:
            case nvinfer1::DataType::kFP8: return 1U;
            case nvinfer1::DataType::kINT4:
                ASSERT(false && "Element size is not implemented for sub-byte data-types.");
        }
        return 0;
    }

    int64_t volume(nvinfer1::Dims const& dims, nvinfer1::Dims const& strides, int32_t vecDim, int32_t comps, int32_t batch)
    {
        int64_t maxNbElems = 1;
        for (int32_t i = 0; i < dims.nbDims; ++i)
        {
            // Get effective length of axis.
            int64_t d = dims.d[i];
            // Any dimension is 0, it is an empty tensor.
            if (d == 0)
            {
                return 0;
            }
            if (i == vecDim)
            {
                d = samplesCommon::divUp(d, comps);
            }
            maxNbElems = std::max(maxNbElems, d * strides.d[i]);
        }
        return maxNbElems * batch * (vecDim < 0 ? 1 : comps);
    }

    nvinfer1::Dims toDims(std::vector<int32_t> const& vec)
    {
        int32_t limit = static_cast<int32_t>(nvinfer1::Dims::MAX_DIMS);
        if (static_cast<int32_t>(vec.size()) > limit)
        {
            sample::gLogWarning << "Vector too long, only first 8 elements are used in dimension." << std::endl;
        }
        // Pick first nvinfer1::Dims::MAX_DIMS elements
        nvinfer1::Dims dims{std::min(static_cast<int32_t>(vec.size()), limit), {}};
        std::copy_n(vec.begin(), dims.nbDims, std::begin(dims.d));
        return dims;
    }

    template <typename T>
    void dumpBuffer(void const* buffer, std::string const& separator, std::ostream& os, Dims const& dims,
        Dims const& strides, int32_t vectorDim, int32_t spv)
    {
        auto const vol = volume(dims);
        T const* typedBuffer = static_cast<T const*>(buffer);
        std::string sep;
        for (int64_t v = 0; v < vol; ++v)
        {
            int64_t curV = v;
            int32_t dataOffset = 0;
            for (int32_t dimIndex = dims.nbDims - 1; dimIndex >= 0; --dimIndex)
            {
                int32_t dimVal = curV % dims.d[dimIndex];
                if (dimIndex == vectorDim)
                {
                    dataOffset += (dimVal / spv) * strides.d[dimIndex] * spv + dimVal % spv;
                }
                else
                {
                    dataOffset += dimVal * strides.d[dimIndex] * (vectorDim == -1 ? 1 : spv);
                }
                curV /= dims.d[dimIndex];
                ASSERT(curV >= 0);
            }

            os << sep;
            sep = separator;
            // print(os, typedBuffer[dataOffset]);
        }
    }


    void loadFromFile(std::string const& fileName, char* dst, size_t size)
    {
        ASSERT(dst);

        std::ifstream file(fileName, std::ios::in | std::ios::binary);
        if (file.is_open())
        {
            file.seekg(0, std::ios::end);
            int64_t fileSize = static_cast<int64_t>(file.tellg());
            // Due to change from int32_t to int64_t VC engines created with earlier versions
            // may expect input of the half of the size
            if (fileSize != static_cast<int64_t>(size) && fileSize != static_cast<int64_t>(size * 2))
            {
                std::ostringstream msg;
                msg << "Unexpected file size for input file: " << fileName << ". Note: Input binding size is: " << size
                    << " bytes but the file size is " << fileSize
                    << " bytes. Double check the size and datatype of the provided data.";
                throw std::invalid_argument(msg.str());
            }
            // Move file pointer back to the beginning after reading file size.
            file.seekg(0, std::ios::beg);
            file.read(dst, size);
            size_t const nbBytesRead = file.gcount();
            file.close();
            if (nbBytesRead != size)
            {
                std::ostringstream msg;
                msg << "Unexpected file size for input file: " << fileName << ". Note: Expected: " << size
                    << " bytes but only read: " << nbBytesRead << " bytes";
                throw std::invalid_argument(msg.str());
            }
        }
        else
        {
            std::ostringstream msg;
            msg << "Cannot open file " << fileName << "!";
            throw std::invalid_argument(msg.str());
        }
    }

    template <typename TType, typename std::enable_if_t<std::is_integral_v<TType>, bool>>
    void fillBuffer(void* buffer, int64_t volume, TType min, TType max)
    {
        TType* typedBuffer = static_cast<TType*>(buffer);
        std::default_random_engine engine;
        std::uniform_int_distribution<int32_t> distribution(min, max);
        auto generator = [&engine, &distribution]() { return static_cast<TType>(distribution(engine)); };
        std::generate(typedBuffer, typedBuffer + volume, generator);
    }

    template <typename TType, typename std::enable_if_t<!std::is_integral_v<TType>, int32_t>>
    void fillBuffer(void* buffer, int64_t volume, TType min, TType max)
    {
        TType* typedBuffer = static_cast<TType*>(buffer);
        std::default_random_engine engine;
        std::uniform_real_distribution<float> distribution(min, max);
        auto generator = [&engine, &distribution]() { return static_cast<TType>(distribution(engine)); };
        std::generate(typedBuffer, typedBuffer + volume, generator);
    }

    // Explicit instantiation
    template void fillBuffer<bool>(void* buffer, int64_t volume, bool min, bool max);
    template void fillBuffer<float>(void* buffer, int64_t volume, float min, float max);
    template void fillBuffer<int32_t>(void* buffer, int64_t volume, int32_t min, int32_t max);
    template void fillBuffer<int64_t>(void* buffer, int64_t volume, int64_t min, int64_t max);
    template void fillBuffer<int8_t>(void* buffer, int64_t volume, int8_t min, int8_t max);
    template void fillBuffer<__half>(void* buffer, int64_t volume, __half min, __half max);
    template void fillBuffer<uint8_t>(void* buffer, int64_t volume, uint8_t min, uint8_t max);


    
std::vector<std::string> splitToStringVec(std::string const& s, char separator, int64_t maxSplit)
{
    std::vector<std::string> splitted;

    for (size_t start = 0; start < s.length();)
    {
        // If maxSplit is specified and we have reached maxSplit, emplace back the rest of the string and break the
        // loop.
        if (maxSplit >= 0 && static_cast<int64_t>(splitted.size()) == maxSplit)
        {
            splitted.emplace_back(s.substr(start, s.length() - start));
            break;
        }

        size_t separatorIndex = s.find(separator, start);
        if (separatorIndex == std::string::npos)
        {
            separatorIndex = s.length();
        }
        splitted.emplace_back(s.substr(start, separatorIndex - start));

        // If the separator is the last character, then we should push an empty string at the end.
        if (separatorIndex == s.length() - 1)
        {
            splitted.emplace_back("");
        }

        start = separatorIndex + 1;
    }

    return splitted;
}
bool matchStringWithOneWildcard(std::string const& pattern, std::string const& target)
{
    auto const splitPattern = splitToStringVec(pattern, '*', 1);

    // If there is no wildcard, return if the two strings match exactly.
    if (splitPattern.size() == 1)
    {
        return pattern == target;
    }

    // Otherwise, target must follow prefix+anything+postfix pattern.
    return target.size() >= (splitPattern[0].size() + splitPattern[1].size()) && target.find(splitPattern[0]) == 0
        && target.rfind(splitPattern[1]) == (target.size() - splitPattern[1].size());
}

// Explicit instantiation
template void dumpBuffer<bool>(void const* buffer, std::string const& separator, std::ostream& os, Dims const& dims,
    Dims const& strides, int32_t vectorDim, int32_t spv);
template void dumpBuffer<int32_t>(void const* buffer, std::string const& separator, std::ostream& os, Dims const& dims,
    Dims const& strides, int32_t vectorDim, int32_t spv);
template void dumpBuffer<int8_t>(void const* buffer, std::string const& separator, std::ostream& os, Dims const& dims,
    Dims const& strides, int32_t vectorDim, int32_t spv);
template void dumpBuffer<float>(void const* buffer, std::string const& separator, std::ostream& os, Dims const& dims,
    Dims const& strides, int32_t vectorDim, int32_t spv);
template void dumpBuffer<__half>(void const* buffer, std::string const& separator, std::ostream& os, Dims const& dims,
    Dims const& strides, int32_t vectorDim, int32_t spv);

template void dumpBuffer<uint8_t>(void const* buffer, std::string const& separator, std::ostream& os, Dims const& dims,
    Dims const& strides, int32_t vectorDim, int32_t spv);
template void dumpBuffer<int64_t>(void const* buffer, std::string const& separator, std::ostream& os, Dims const& dims,
    Dims const& strides, int32_t vectorDim, int32_t spv);
}