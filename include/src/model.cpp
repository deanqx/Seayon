#include "../seayon.hpp"
#include <stdio.h>

size_t seayon::model::save(std::vector<char>& buffer) const
{
    size_t buffersize{};

    const uint8_t _printloss = (uint8_t)printloss;
    const int32_t _seed = (int32_t)seed;
    const uint32_t _layerCount = layers.size();
    const uint32_t _logLenght = logfolder.size();
    std::vector<uint32_t> layout(layers.size());
    std::vector<ActivFunc> a(layers.size() - 1);
    uint32_t parameters_size = sizeof(uint8_t) + sizeof(int32_t) + (layers.size() + 2) * sizeof(uint32_t) + a.size() * sizeof(ActivFunc) + _logLenght * sizeof(char);

    std::vector<size_t> nSize(layers.size());
    std::vector<size_t> wSize(layers.size());
    for (int i = 0; i < layers.size(); ++i)
    {
        layout[i] = (uint32_t)layers[i].nCount;
        if (i > 0)
        {
            a[i - 1] = layers[i].func;

            nSize[i] = layers[i].nCount * sizeof(float);
            wSize[i] = layers[i].wCount * sizeof(float);
            buffersize += nSize[i] + wSize[i];
        }
    }

    buffersize += sizeof(uint32_t) + parameters_size;
    buffer.resize(buffersize);

    char* pointer = buffer.data();

    memcpy(buffer.data(), &parameters_size, sizeof(uint32_t));
    pointer += sizeof(uint32_t);

    memcpy(pointer, &_printloss, sizeof(uint8_t));
    pointer += sizeof(uint8_t);
    memcpy(pointer, &_seed, sizeof(int32_t));
    pointer += sizeof(int32_t);

    memcpy(pointer, &_layerCount, sizeof(uint32_t));
    pointer += sizeof(uint32_t);
    memcpy(pointer, &_logLenght, sizeof(uint32_t));
    pointer += sizeof(uint32_t);

    memcpy(pointer, layout.data(), layers.size() * sizeof(uint32_t));
    pointer += layers.size() * sizeof(uint32_t);
    memcpy(pointer, a.data(), a.size() * sizeof(ActivFunc));
    pointer += a.size() * sizeof(ActivFunc);
    memcpy(pointer, logfolder.data(), _logLenght * sizeof(char));
    pointer += _logLenght * sizeof(char);

    for (int i = 1; i < layers.size(); ++i)
    {
        memcpy(pointer, layers[i].weights.data(), wSize[i]);
        pointer += wSize[i];
        memcpy(pointer, layers[i].biases.data(), nSize[i]);
        pointer += nSize[i];
    }

    return buffersize;
}
size_t seayon::model::save_file(const char* path) const
{
    std::vector<char> buffer;
    size_t buffersize = save(buffer);

    FILE* file = fopen(path, "wb");

    if (!file)
    {
        return 0;
    }

    fwrite(buffer.data(), sizeof(char), buffersize, file);
    fclose(file);

    return buffersize;
}
void seayon::model::load(const char* buffer)
{
    std::vector<size_t> nSize(layers.size());
    std::vector<size_t> wSize(layers.size());
    for (int i = 1; i < layers.size(); ++i)
    {
        nSize[i] = layers[i].nCount * sizeof(float); // WARN float is not const size
        wSize[i] = layers[i].wCount * sizeof(float);
    }

    const char* pointer = buffer;

    uint32_t parameters_size{};

    memcpy(&parameters_size, pointer, sizeof(uint32_t));
    pointer += sizeof(uint32_t) + parameters_size;

    for (int i = 1; i < layers.size(); ++i)
    {
        memcpy(layers[i].weights.data(), pointer, wSize[i]);
        pointer += wSize[i];
        memcpy(layers[i].biases.data(), pointer, nSize[i]);
        pointer += nSize[i];
    }
}
bool seayon::model::load_file(const char* path)
{
    FILE* file = fopen(path, "rb");
    if (!file)
    {
        return true;
    }

    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);

    std::vector<char> buffer(size);
    fread(buffer.data(), sizeof(char), size, file);
    fclose(file);

    load(buffer.data());

    return false;
}
void seayon::model::copy(model& to) const
{
    for (int l = 1; l < layers.size(); ++l)
    {
        memcpy(to.layers[l].biases.data(), layers[l].biases.data(), layers[l].nCount * sizeof(float));
        memcpy(to.layers[l].weights.data(), layers[l].weights.data(), layers[l].wCount * sizeof(float));
    }
}
void seayon::model::combine_into(model** with, int count)
{
    for (int l2 = 1; l2 < layers.size(); ++l2)
    {
        const int l1 = l2 - 1;

        for (int n2 = 0; n2 < layers[l2].nCount; ++n2)
        {
            float ab = 0.0f;
            for (int i = 0; i < count; ++i)
                ab += with[i]->layers[l2].biases[n2];

            layers[l2].biases[n2] = ab / count;

            for (int n1 = 0; n1 < layers[l1].nCount; ++n1)
            {
                float aw = 0.0f;
                for (int i = 0; i < count; ++i)
                    aw += with[i]->layers[l2].weights[n2 * layers[l1].nCount + n1];

                layers[l2].weights[n2 * layers[l1].nCount + n1] = aw / count;
            }
        }
    }
}
bool seayon::model::equals(model& second)
{
    bool equal = true;

    for (int i = 1; equal && i < layers.size(); ++i)
    {
        for (int w = 0; equal && w < layers[i].wCount; ++w)
            equal = (layers[i].weights[w] == second.layers[i].weights[w]);

        for (int n = 0; equal && n < layers[i].nCount; ++n)
            equal = (layers[i].biases[n] == second.layers[i].biases[n]);
    }

    return equal;
}

std::vector<float> seayon::model::denormalized(const float max, const float min) const
{
    const layer& last = layers[layers.size() - 1];
    const float range = max - min;

    std::vector<float> out(last.nCount);

    for (int i = 0; i < last.nCount; ++i)
    {
        out[i] = last.neurons[i] * range + min;
    }

    return out;
}

bool seayon::model::check(const dataset& data) const
{
    return layers[0].nCount == xsize && layers[layers.size() - 1].nCount == ysize;
}