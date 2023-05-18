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

bool ImportMnist(const int sampleCount, seayon::dataset& data, const std::string file_without_extension, int thread_count = 1)
{
    const int per_thread = sampleCount / thread_count;

    auto start = std::chrono::high_resolution_clock::now();

    printf("\tLoading mnist...");

    std::string path(file_without_extension + ".csv");
    FILE* file = fopen(path.c_str(), "r");
    if (file == nullptr)
    {
        printf("Cannot open csv file\n");
        return false;
    }

    data.reserve(sampleCount);

    constexpr int formatLenght = 785 * 3 - 1;
    char format[formatLenght];
    format[formatLenght - 2] = '%';
    format[formatLenght - 3] = 'd';
    for (int i = 0; i < formatLenght - 2; i += 3)
    {
        format[i] = '%';
        format[i + 1] = 'd';
        format[i + 2] = ',';
    }

    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);

    std::vector<char> buffer(size);

    fread(buffer.data(), sizeof(char), size, file);

    size_t pos = 0;
    for (int i = 0; i < sampleCount; ++i)
    {
        memset(data[i].y, 0, 10 * sizeof(float));

        data[i].y[parse_value(buffer.data(), pos)] = 1.0f;

        for (int k = 0; k < 784; ++k)
        {
            const int val = parse_value(buffer.data(), pos);
            if (val == 0)
                data[i].x[k] = 0;
            else
                data[i].x[k] = (float)val / 255.0f;
        }
    }
    fclose(file);

    printf("DONE (%ims)\n", (int)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count());

    return true;
}