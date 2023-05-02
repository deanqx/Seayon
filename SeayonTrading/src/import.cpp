#include <vector>
#include <string>
#include <sstream>
#include "seayon.hpp"

float parse_line(const std::string& line)
{
    int pos = 0;

    for (int column = 0; pos < line.size() && column < 3; ++pos)
        if (line[pos] == '\t')
            ++column;

    std::stringstream high;
    for (; pos < line.size() && line[pos] != '\t'; ++pos)
        high << line[pos];

    return std::stof(high.str());
}

template <int INPUTS, int OUTPUTS>
void parse_sample(const std::vector<std::string>& lines, const int begin, int end, typename trainingdata<INPUTS, OUTPUTS>::sample& sample)
{
    float origin = parse_line(lines[end]);
    int before = 0;
    int after = 0;

    for (int i = begin; i <= end; ++i, ++before)
    {
        sample.inputs[before] = parse_line(lines[i]) / origin;
    }

    end += OUTPUTS + 1;

    for (int i = end - OUTPUTS + 1; i <= end; ++i, ++after)
    {
        sample.outputs[after] = parse_line(lines[i]) / origin;
    }
}

template <int INPUTS, int OUTPUTS>
bool reimport(int sampleCount, trainingdata<INPUTS, OUTPUTS>& data, const std::string file_without_extension, int thread_count = 1)
{
    auto start = std::chrono::high_resolution_clock::now();
    printf("\tLoading dataset...");

    std::ifstream csv(file_without_extension + ".csv");

    if (!csv.is_open())
    {
        std::cout << "Cannot open csv file: " << file_without_extension + ".csv" << "\n";
        return false;
    }

    std::ofstream new_binary(file_without_extension + ".bin", std::ios::binary);

    std::vector<std::string> lines;
    std::vector<std::thread> threads(thread_count);

    lines.reserve(std::abs(sampleCount));

    std::string garbage;
    std::getline(csv, garbage);
    for (int i = 0; ; ++i)
    {
        std::string line;
        if (!std::getline(csv, line))
            break;

        lines.push_back(line);
    }

    if (sampleCount < 1)
        sampleCount = lines.size() - INPUTS - OUTPUTS - 1;

    if (sampleCount < 1)
    {
        printf("error: dataset is too small\t");
        return false;
    }

    const int per_thread = sampleCount / thread_count;
    data.reserve(sampleCount);

    for (int t = 0; t < thread_count; ++t)
    {
        threads[t] = std::thread([&, t]
            {
                const int begin = t * per_thread;
                const int end = begin + per_thread - 1;

                for (int i = begin; i <= end; ++i)
                {
                    parse_sample<INPUTS, OUTPUTS>(lines, i, i + INPUTS - 1, data[i]);
                }
            });
    }

    const int begin = thread_count * per_thread;
    const int end = sampleCount - 1;
    for (int i = begin; i <= end; ++i)
    {
        parse_sample<INPUTS, OUTPUTS>(lines, i, i + INPUTS - 1, data[i]);
    }

    for (int t = 0; t < thread_count; ++t)
    {
        threads[t].join();
    }


    new_binary.write((char*)&data[0], sampleCount * sizeof(typename trainingdata<INPUTS, OUTPUTS>::sample));
    // new_binary.flush();

    // trainingdata<INPUTS, OUTPUTS> comp;
    // import(sampleCount, comp, file_without_extension);

    // for (int i = 0; i < sampleCount; ++i)
    // {
    //     if (std::memcmp(data[i], comp[i], sizeof(trainingdata<INPUTS, OUTPUTS>::sample)))
    //         printf("pos %i\n", i);
    // }

    printf("DONE (%ims) %i samples\n", (int)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count(), sampleCount);
    return true;
}

template <int INPUTS, int OUTPUTS>
bool import(const int sampleCount, trainingdata<INPUTS, OUTPUTS>& data, const std::string file_without_extension, int thread_count = 1)
{
    if (sampleCount < 1)
    {
        printf("error: sampleCount is too small\t");
        return false;
    }

    std::ifstream binary(file_without_extension + ".bin", std::ios::binary);

    if (binary.is_open())
    {
        auto start = std::chrono::high_resolution_clock::now();
        printf("\tLoading dataset...");

        data.reserve(sampleCount);
        binary.read((char*)&data[0], sampleCount * sizeof(typename trainingdata<INPUTS, OUTPUTS>::sample));

        printf("DONE (%ims)\n", (int)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count());
        return true;
    }
    else
    {
        return reimport(sampleCount, data, file_without_extension, thread_count);
    }
}