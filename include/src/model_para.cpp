#include "../seayon.hpp"
#include <stdio.h>

void seayon::model_parameters::load_parameters(const char* buffer)
{
    const char* pointer = buffer + sizeof(uint32_t);

    memcpy(&seed, pointer, sizeof(int32_t));
    pointer += sizeof(int32_t);

    uint32_t layerCount;
    uint32_t logLenght;

    memcpy(&layerCount, pointer, sizeof(uint32_t));
    pointer += sizeof(uint32_t);
    memcpy(&logLenght, pointer, sizeof(uint32_t));
    pointer += sizeof(uint32_t);

    layout.resize(layerCount);
    a.resize(layerCount - 1);
    logfolder.resize(logLenght);

    std::vector<uint32_t> _layout(layerCount);

    memcpy(_layout.data(), pointer, layerCount * sizeof(uint32_t));
    pointer += layerCount * sizeof(uint32_t);
    memcpy(a.data(), pointer, a.size() * sizeof(ActivFunc));
    pointer += a.size() * sizeof(ActivFunc);
    memcpy((void*)logfolder.data(), pointer, logLenght * sizeof(char));
    // pointer += logLenght * sizeof(char);

    for (int i = 0; i < layerCount; ++i)
    {
        layout[i] = (int)_layout[i];
    }
}

bool seayon::model_parameters::load_parameters_file(const char* path)
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

    load_parameters(buffer.data());

    return false;
}