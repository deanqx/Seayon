#include "../seayon.hpp"
#include <stdio.h>

seayon::dataset::dataset(const int inputSize, const int outputSize) : xsize(inputSize), ysize(outputSize)
{
}

seayon::dataset::dataset(const int inputSize, const int outputSize, const int newsize) : xsize(inputSize), ysize(outputSize)
{
    resize(newsize);
}

seayon::dataset::dataset(std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& outputs)
    : dataset((int)inputs[0].size(), (int)outputs[0].size(), inputs.size())
{
    if (inputs.size() != outputs.size())
    {
        printf("--- error: inputs and outputs are not equal size ---\n");
        return;
    }

    for (int i = 0; i < inputs.size(); ++i)
    {
        samples[i].x.swap(inputs[i]);
        samples[i].y.swap(outputs[i]);
    }
}

void seayon::dataset::resize(const int newsize)
{
    if (samples.size() < newsize)
    {
        samples.reserve(newsize);

        for (int i = samples.size(); i < newsize; ++i)
        {
            samples.emplace_back(xsize, ysize);
        }
    }
    else
    {
        samples.erase(samples.end() - newsize, samples.end()); // TODO test
    }
}

inline const seayon::dataset::sample& seayon::dataset::operator[](const int index) const
{
    return samples[index];
}

size_t seayon::dataset::save(std::vector<char>& out_buffer) const
{
    size_t size = sizeof(int32_t) + samples.size() * xsize * sizeof(float) + samples.size() * ysize * sizeof(float);

    out_buffer.resize(size);

    char* pointer = out_buffer.data();

    int32_t sampleCount = samples.size();
    memcpy(pointer, &sampleCount, sizeof(int32_t));
    pointer += sizeof(int32_t);

    for (int i = 0; i < samples.size(); ++i)
    {
        memcpy(pointer, samples[i].x.data(), xsize * sizeof(float));
        pointer += xsize * sizeof(float);
        memcpy(pointer, samples[i].y.data(), ysize * sizeof(float));
        pointer += ysize * sizeof(float);
    }

    return size;
}

size_t seayon::dataset::save_file(const char* path) const
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

void seayon::dataset::load(const char* buffer)
{
    const char* pointer = buffer;

    int32_t sampleCount{};
    memcpy(&sampleCount, pointer, sizeof(int32_t));
    pointer += sizeof(int32_t);

    resize(sampleCount);

    for (int i = 0; i < samples.size(); ++i)
    {
        memcpy(samples[i].x.data(), pointer, xsize * sizeof(float));
        pointer += xsize * sizeof(float);
        memcpy(samples[i].y.data(), pointer, ysize * sizeof(float));
        pointer += ysize * sizeof(float);
    }
}
bool seayon::dataset::load_file(const char* path)
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

float seayon::dataset::max_value() const
{
    float max = samples[0].x[0];

    for (int s = 0; s < samples.size(); ++s)
    {
        for (int i = 0; i < xsize; ++i)
        {
            const float& x = samples[s].x[i];

            if (x > max)
                max = x;
        }

        for (int i = 0; i < ysize; ++i)
        {
            const float& x = samples[s].y[i];

            if (x > max)
                max = x;
        }
    }

    return max;
}

float seayon::dataset::min_value() const
{
    float min = samples[0].x[0];

    for (int s = 0; s < samples.size(); ++s)
    {
        for (int i = 0; i < xsize; ++i)
        {
            const float& x = samples[s].x[i];

            if (x < min)
                min = x;
        }

        for (int i = 0; i < ysize; ++i)
        {
            const float& x = samples[s].y[i];

            if (x < min)
                min = x;
        }
    }

    return min;
}

void seayon::dataset::normalize(const float max, const float min)
{
    const float range = max - min;

    for (int i = 0; i < samples.size(); ++i)
    {
        for (int in = 0; in < xsize; ++in)
        {
            samples[i].x[in] = (samples[i].x[in] - min) / range;
        }

        for (int out = 0; out < ysize; ++out)
        {
            samples[i].y[out] = (samples[i].y[out] - min) / range;
        }
    }
}

void seayon::dataset::shuffle()
{
    std::random_device rm_seed;
    std::shuffle(samples.begin(), samples.end(), std::mt19937(rm_seed()));
}