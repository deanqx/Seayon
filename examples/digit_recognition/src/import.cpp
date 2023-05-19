#include <vector>
#include <sstream>
#include <chrono>
#include <thread>
#include "seayon.hpp"

int parse_value(char* buffer, size_t& pos)
{
    char extracted[3];

    if (buffer[pos] == '0')
    {
        pos += 2;
        return 0;
    }

    extracted[0] = buffer[pos];
    ++pos;
    if (buffer[pos] != ',')
    {
        extracted[1] = buffer[pos];
        ++pos;
        if (buffer[pos] != ',')
        {
            extracted[2] = buffer[pos];
            ++pos;
        }
        else
            extracted[2] = 0;
    }
    else
    {
        extracted[1] = 0;
        extracted[2] = 0;
    }
    ++pos;

    return atoi(extracted);
}

bool ImportMnist(const int sampleCount, seayon::dataset& data, const std::string file_without_extension)
{
    auto start = std::chrono::high_resolution_clock::now();

    printf("Loading mnist...");

    std::string path(file_without_extension + ".csv");
    FILE* file = fopen(path.c_str(), "r");
    if (file == nullptr)
    {
        printf("Cannot open csv file\n");
        return false;
    }

    data.resize(sampleCount);

    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);

    std::vector<char> buffer(size);

    fread(buffer.data(), sizeof(char), size, file);

    size_t pos = 0;
    for (int i = 0; i < sampleCount; ++i)
    {
        data.samples[i].y[parse_value(buffer.data(), pos)] = 1.0f;

        for (int k = 0; k < 784; ++k)
        {
            const int val = parse_value(buffer.data(), pos);
            if (val == 0)
                data.samples[i].x[k] = 0.0f;
            else
                data.samples[i].x[k] = (float)val / 255.0f;
        }
    }
    fclose(file);

    printf("DONE (%ims)\n", (int)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count());

    return true;
}