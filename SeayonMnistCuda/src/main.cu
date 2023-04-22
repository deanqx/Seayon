#include <sstream>
#include <iostream>
#include <chrono>
#include <vector>
#include "../../include/cuda_seayon.cuh"
#include "../../SeayonMnist/src/import.cpp"

int main()
{
    constexpr bool load = false;
    constexpr bool printcost = true;

    constexpr int runCount = 1;
    constexpr float learningRate = 0.00003f;
    constexpr float momentum = 0.00001f;
    constexpr int batch_size = 50;

    std::vector<int> layout = { 784, 16, 16, 10 };
    std::vector<ActivFunc> funcs = { ActivFunc::SIGMOID, ActivFunc::SIGMOID, ActivFunc::SIGMOID, ActivFunc::SIGMOID };
    cuda_seayon nn(layout, funcs, 1472, printcost, "../../../../SeayonMnist/res/logs");
    seayon nn2(layout, funcs, 1472, printcost, "../../../../SeayonMnist/res/logs");

    trainingdata<784, 10> testdata;

    std::ifstream test("../../../../SeayonMnist/res/mnist/mnist_test.csv");
    if (ImportMnist(10000, testdata, test, 1))
        return 1;
    test.close();

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

        std::ifstream train("../../../../SeayonMnist/res/mnist/mnist_train.csv");
        if (train.is_open() && true)
        {
            trainingdata<784, 10> traindata;
            if (ImportMnist(60000, traindata, train, 1))
                return 1;

            printf("\n");
            nn.printo(traindata, 0);

            auto start1 = std::chrono::high_resolution_clock::now();
            nn.fit(runCount, traindata, testdata, ParallelOptimizer::MINI_BATCH, learningRate, momentum, batch_size);
            auto end1 = std::chrono::high_resolution_clock::now();

            auto start2 = std::chrono::high_resolution_clock::now();
            nn2.fit(runCount, traindata, testdata, Optimizer::MINI_BATCH, learningRate, momentum, batch_size, 32);
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

            auto start1 = std::chrono::high_resolution_clock::now();
            nn.fit(runCount, testdata, testdata, ParallelOptimizer::MINI_BATCH, learningRate, momentum, batch_size);
            auto end1 = std::chrono::high_resolution_clock::now();

            auto start2 = std::chrono::high_resolution_clock::now();
            nn2.fit(runCount, testdata, testdata, Optimizer::MINI_BATCH, learningRate, momentum, batch_size, 32);
            auto end2 = std::chrono::high_resolution_clock::now();

            auto time1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
            auto time2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);

            printf("Cuda:   %lldms\n", time1.count());
            printf("Native: %lldms\n", time2.count());

            nn.printo(testdata, 0);
        }

        std::ofstream file("../../../../SeayonMnist/res/mnist.bin", std::ios::binary);
        nn.save(file);
    }

    nn.printo(testdata, 0);

    return 0;
}