#include "../seayon.hpp"

void seayon::model_parameters::load_parameters(const char* buffer)
{
    const char* pointer = buffer + sizeof(uint32_t);

    memcpy(&printloss, pointer, sizeof(uint8_t));
    pointer += sizeof(uint8_t);
    memcpy(&seed, pointer, sizeof(int32_t));
    pointer += sizeof(int32_t);

    uint32_t sampleCount{};
    uint32_t logLenght{};

    memcpy(&sampleCount, pointer, sizeof(uint32_t));
    pointer += sizeof(uint32_t);
    memcpy(&logLenght, pointer, sizeof(uint32_t));
    pointer += sizeof(uint32_t);

    layout.resize(sampleCount);
    a.resize(sampleCount - 1);
    logfolder.resize(logLenght);

    std::vector<uint32_t> _layout(sampleCount);

    memcpy(_layout.data(), pointer, sampleCount * sizeof(uint32_t));
    pointer += sampleCount * sizeof(uint32_t);
    memcpy(a.data(), pointer, a.size() * sizeof(ActivFunc));
    pointer += a.size() * sizeof(ActivFunc);
    memcpy((void*)logfolder.data(), pointer, logLenght * sizeof(char));
    // pointer += logLenght * sizeof(char);

    for (int i = 0; i < sampleCount; ++i)
    {
        layout[i] = (int)_layout[i];
    }
}

bool seayon::model_parameters::load_parameters(std::ifstream& file)
{
    if (file.is_open())
    {
        file.seekg(0, file.end);
        int N = (int)file.tellg();
        file.seekg(0, file.beg);

        std::vector<char> buffer(N);
        file.read(buffer.data(), N);
        load_parameters(buffer.data());

        return true;
    }

    return false;
}