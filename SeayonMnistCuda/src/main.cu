#include <sstream>
#include <iostream>
#include <chrono>
#include <vector>
#include "../../include/cuda_seayon.cuh"
#include "../../SeayonMnist/src/import.cpp"

int main()
{
    constexpr bool load = false;
    constexpr bool printloss = true;

    constexpr int runCount = 1;
    constexpr float learningRate = 0.03f;
    constexpr float momentum = 0.5f;
    constexpr int thread_count = 32;

    std::vector<int> layout = { 784, 16, 16, 10 };
    std::vector<ActivFunc> funcs = { ActivFunc::RELU, ActivFunc::SIGMOID, ActivFunc::SIGMOID, ActivFunc::SIGMOID };
    cuda_seayon nn(layout, funcs, 1472, printloss, "../../../../SeayonMnist/res/logs");
    seayon nn2(layout, funcs, 1472, printloss, "../../../../SeayonMnist/res/logs");

    trainingdata<784, 10> testdata;

    if (!ImportMnist(10000, testdata, "../../../../SeayonMnist/res/mnist/mnist_test"))
        return 1;

    if (load)
    {
        std::ifstream file("../../../../SeayonMnist/res/mnist.bin", std::ios::binary);
        nn.load(file);
    }
    else
    {
        // !!! You should download the full mnist-set !!!
        // 1. Download mnist_train.csv (from for example https://yann.lecun.com)
        // 2. Copy the header(first line) from mnist_test.csv
        // 3. Put mnist_train.csv in the "res/mnist/" folder

        const std::string traindata_path("../../../../SeayonMnist/res/mnist/mnist_train");
        std::ifstream exists(traindata_path + ".csv");
        if (exists.good() && true)
        {
            trainingdata<784, 10> traindata;
            if (!ImportMnist(60000, traindata, traindata_path))
                return 1;

            printf("\n");
            nn.printo(traindata, 0);

            auto start1 = std::chrono::high_resolution_clock::now();
            nn.fit(runCount, traindata, testdata, ParallelOptimizer::MINI_BATCH, learningRate, momentum, thread_count);
            auto end1 = std::chrono::high_resolution_clock::now();

            auto start2 = std::chrono::high_resolution_clock::now();
            nn2.fit(runCount, traindata, testdata, Optimizer::MINI_BATCH, learningRate, momentum, 32);
            auto end2 = std::chrono::high_resolution_clock::now();

            auto time1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
            auto time2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);

            printf("Cuda:   %lldms\n", time1.count());
            printf("Native: %lldms\n", time2.count());

            nn.printo(traindata, 0);
        }
        else
        {
            printf("\n");
            nn.printo(testdata, 0);
            nn.fit(runCount, testdata, testdata, ParallelOptimizer::MINI_BATCH, learningRate, momentum, thread_count);
            nn.printo(testdata, 0);
        }

        std::ofstream file("../../../../SeayonMnist/res/mnist.bin", std::ios::binary);
        nn.save(file);
    }

    return 0;
}