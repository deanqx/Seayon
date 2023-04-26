#include <vector>
#include <sstream>
#include <chrono>
#include <thread>
#include "seayon.hpp"

static void parse_line(const std::string* lines, const int begin, const int end, trainingdata<784, 10>& data)
{
    // auto start = std::chrono::high_resolution_clock::now();

    for (int i = begin; i <= end; ++i)
    {
        int pos = 0;
        auto& sample = data[i];

        std::stringstream label;
        for (; pos < lines[i].size() && lines[i][pos] != ','; ++pos)
            label << lines[i][pos];

        ++pos;

        sample.outputs[stoi(label.str())] = 1.0f;

        for (int pixelIndex = 0; pixelIndex < 784; ++pixelIndex)
        {
            std::stringstream pixel;
            for (; pos < lines[i].size() && lines[i][pos] != ','; ++pos)
                pixel << lines[i][pos];

            ++pos;

            sample.inputs[pixelIndex] = (float)stoi(pixel.str()) / 255.0f;
        }
    }

    // printf("FINISHED[%i -> %i(%i)] (%ims)\n", begin, end, end + 1 - begin, (int)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count());
}

bool ImportMnist(const int sampleCount, trainingdata<784, 10>& data, const std::string file_without_extension, int thread_count = 1)
{
    const int per_thread = sampleCount / thread_count;

    auto start = std::chrono::high_resolution_clock::now();

    printf("\tLoading mnist...");

    std::ifstream binary(file_without_extension + ".bin", std::ios::binary);

    if (binary.is_open())
    {
        data.reserve(sampleCount);
        binary.read((char*)&data[0], sampleCount * sizeof(typename trainingdata<784, 10>::sample));
    }
    else
    {
        std::ifstream csv(file_without_extension + ".csv");
        std::ofstream new_binary(file_without_extension + ".bin", std::ios::binary);

        if (!csv.is_open())
        {
            printf("Cannot open csv file\n");
            return false;
        }

        data.reserve(sampleCount);
        std::vector<std::string> lines(sampleCount);
        std::vector<std::thread> threads(thread_count);

        for (int i = 0; i < sampleCount; ++i)
        {
            std::getline(csv, lines[i]);
        }

        for (int t = 0; t < thread_count; ++t)
        {
            threads[t] = std::thread([&, t]
                {
                    const int begin = t * per_thread;
                    const int end = begin + per_thread - 1;
                    parse_line(lines.data(), begin, end, data);
                });
        }

        const int begin = thread_count * per_thread;
        const int end = sampleCount - 1;
        parse_line(lines.data(), begin, end, data);

        for (int t = 0; t < thread_count; ++t)
        {
            threads[t].join();
        }

        new_binary.write((char*)&data[0], sampleCount * sizeof(typename trainingdata<784, 10>::sample));
    }

    printf("DONE (%ims)\n", (int)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count());

    return true;
}