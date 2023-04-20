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

    constexpr int runCount = 1000;
    constexpr float learningRate = 0.00003f;
    constexpr float momentum = 0.00001f;
    constexpr int batch_size = 10;

    std::vector<int> layout = { 784, 16, 16, 10 };
    std::vector<ActivFunc> funcs = { ActivFunc::SIGMOID, ActivFunc::SIGMOID, ActivFunc::SIGMOID, ActivFunc::SIGMOID };
    cuda_seayon nn(layout, funcs, printcost, 1472, "../../../../SeayonMnist/res/logs");

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
            nn.fit(runCount, traindata, testdata, ParallelOptimizer::MINI_BATCH, learningRate, momentum, batch_size);
            nn.printo(traindata, 0);
        }
        else
        {
            printf("\n");
            nn.printo(testdata, 0);
            nn.fit(runCount, testdata, testdata, ParallelOptimizer::MINI_BATCH, learningRate, momentum, batch_size);
            nn.printo(testdata, 0);
        }

        std::ofstream file("../../../../SeayonMnist/res/mnist.bin", std::ios::binary);
        nn.save(file);
    }

    return 0;
}