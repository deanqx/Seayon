#include <iostream>
#include <fstream>
#include "seayon.hpp"
#include "import.cpp"

int main()
{
	constexpr bool load = false;
	constexpr bool printloss = true;

	constexpr int runCount = 50;
	constexpr float learningRate = 0.03f;
	constexpr float momentum = 0.5f;
	constexpr int thread_count = 32;

	std::vector<int> layout = { 784, 16, 16, 10 };
	std::vector<ActivFunc> funcs = { ActivFunc::SIGMOID, ActivFunc::SIGMOID, ActivFunc::SIGMOID, ActivFunc::SIGMOID };
	seayon nn(layout, funcs, 1472, printloss, "../../../../SeayonMnist/res/logs");

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
		if (exists.good() && false)
		{
			trainingdata<784, 10> traindata;
			if (!ImportMnist(60000, traindata, traindata_path))
				return 1;

			printf("\n");
			nn.printo(traindata, 0);
			nn.fit(runCount, traindata, testdata, Optimizer::MINI_BATCH, learningRate, momentum, thread_count);
			nn.printo(traindata, 0);
		}
		else
		{
			printf("\n");
			nn.printo(testdata, 0);
			nn.fit(runCount, testdata, testdata, Optimizer::STOCHASTIC, learningRate, momentum, thread_count);
			nn.printo(testdata, 0);
		}

		std::ofstream file("../../../../SeayonMnist/res/mnist.bin", std::ios::binary);
		nn.save(file);
	}

	float acc = nn.printo(testdata, 0);

	if (acc < 0.5f) // (unit test) Has it learned slightly
		return 2;

	return 0;
}