#include "../seayon.hpp"

void seayon::dataset::sample::clear()
{
    delete[] x;
    delete[] y;
}

seayon::dataset::dataset(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& outputs)
    : dataset((int)inputs[0].size(), (int)outputs[0].size(), (int)inputs.size())
{
    if (inputs.size() != outputs.size())
    {
        printf("--- error: inputs and outputs are not equal size ---\n");
        return;
    }

    for (int i = 0; i < sampleCount; ++i)
    {
        for (int k = 0; k < xsize; ++k)
        {
            samples[i].x[k] = inputs[i][k];
        }

        for (int k = 0; k < ysize; ++k)
        {
            samples[i].y[k] = outputs[i][k];
        }
    }
}

void seayon::dataset::reserve(const int reserved)
{
    clear();

    sampleCount = reserved;
    samples = (sample*)malloc(sampleCount * sizeof(sample));

    for (int i = 0; i < sampleCount; ++i)
    {
        new (&samples[i]) sample(xsize, ysize);
    }
}

int seayon::dataset::size() const
{
    return sampleCount;
}

seayon::dataset::sample& seayon::dataset::operator[](const int i) const
{
    return samples[i];
}

seayon::dataset::sample* seayon::dataset::get(const int i) const
{
    return samples + i * sizeof(sample);
}

size_t seayon::dataset::save(std::vector<char>& out_buffer)
{
    size_t size = sizeof(int32_t) + sampleCount * xsize * sizeof(float) + sampleCount * ysize * sizeof(float);

    out_buffer.resize(size);

    char* pointer = out_buffer.data();

    int32_t _sampleCount = sampleCount;
    memcpy(pointer, &_sampleCount, sizeof(int32_t));
    pointer += sizeof(int32_t);

    for (int i = 0; i < sampleCount; ++i)
    {
        memcpy(pointer, samples[i].x, xsize * sizeof(float));
        pointer += xsize * sizeof(float);
        memcpy(pointer, samples[i].y, ysize * sizeof(float));
        pointer += ysize * sizeof(float);
    }

    return size;
}

size_t seayon::dataset::save(std::ofstream& file)
{
    std::vector<char> buffer;
    size_t buffersize = save(buffer);

    file.write(buffer.data(), buffersize);
    if (file.fail())
        buffersize = 0;

    file.flush();

    return buffersize;
}

void seayon::dataset::load(const char* buffer)
{
    const char* pointer = buffer;

    int32_t _sampleCount{};
    memcpy(&_sampleCount, pointer, sizeof(int32_t));
    pointer += sizeof(int32_t);

    sampleCount = (int)_sampleCount;
    reserve(sampleCount);

    for (int i = 0; i < sampleCount; ++i)
    {
        memcpy(samples[i].x, pointer, xsize * sizeof(float));
        pointer += xsize * sizeof(float);
        memcpy(samples[i].y, pointer, ysize * sizeof(float));
        pointer += ysize * sizeof(float);
    }
}
bool seayon::dataset::load(std::ifstream& file)
{
    if (file.is_open())
    {
        file.seekg(0, file.end);
        int N = (int)file.tellg();
        file.seekg(0, file.beg);

        std::vector<char> buffer(N);
        file.read(buffer.data(), N);
        load(buffer.data());

        return true;
    }

    return false;
}

void seayon::dataset::swap(dataset* with)
{
    sample* const p = samples;
    const int n = sampleCount;

    samples = with->samples;
    sampleCount = with->sampleCount;
    with->samples = samples;
    with->sampleCount = sampleCount;
}

float seayon::dataset::max_value() const
{
    float max = samples[0].x[0];

    for (int s = 0; s < sampleCount; ++s)
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

    for (int s = 0; s < sampleCount; ++s)
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

    for (int i = 0; i < sampleCount; ++i)
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
    // TODO
    // std::random_device rm_seed;
    // std::shuffle(samples, samples + sampleCount - 1, std::mt19937(rm_seed()));
}

void seayon::dataset::clear()
{
    if (manageMemory)
    {
        if (samples != nullptr)
        {
            for (int i = 0; i < sampleCount; ++i)
            {
                samples[i].clear();
            }
            free(samples);
        }
    }
}

seayon::dataset::~dataset()
{
    clear();
}