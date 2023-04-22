#include <iostream>
#include <fstream>
#include "seayon.hpp"
#include "import.cpp"

int main()
{
	constexpr bool load = false;
	constexpr bool printcost = true;

	constexpr int runCount = 10000;
	constexpr float learningRate = 0.00003f;
	constexpr float momentum = 0.00001f;
	// constexpr float learningRate = 0.03f;
	// constexpr float momentum = 0.1f;
	constexpr int batch_size = 100;
	constexpr int thread_count = 32;

	std::vector<int> layout = { 784, 16, 16, 10 };
	std::vector<ActivFunc> funcs = { ActivFunc::SIGMOID, ActivFunc::SIGMOID, ActivFunc::SIGMOID, ActivFunc::SIGMOID };
	seayon nn(layout, funcs, 1472, printcost, "../../../../SeayonMnist/res/logs");

	trainingdata<784, 10> testdata;

	std::ifstream test("../../../../SeayonMnist/res/mnist/mnist_test.csv");
	if (ImportMnist(10000, testdata, test))
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
			if (ImportMnist(60000, traindata, train))
				return 1;

			printf("\n");
			nn.printo(traindata, 0);
			nn.fit(runCount, traindata, testdata, Optimizer::MINI_BATCH, learningRate, momentum, batch_size, thread_count);
			nn.printo(traindata, 0);
		}
		else
		{
			printf("\n");
			nn.printo(testdata, 0);
			nn.fit(runCount, testdata, testdata, Optimizer::MINI_BATCH, learningRate, momentum, batch_size, thread_count);
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