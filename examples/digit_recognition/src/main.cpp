#include <iostream>
#include <fstream>
#include "seayon.hpp"
#include "import.cpp"

using namespace seayon;

int main()
{
	constexpr bool load = false;
	constexpr bool printloss = true;

	constexpr int runCount = 100;
	constexpr float learning_rate = 0.001f;
	constexpr float momentum = 0.9f;
	constexpr int batch_size = 1;
	constexpr int thread_count = 10;

	std::vector<int> layout = { 784, 16, 16, 10 };
	std::vector<ActivFunc> funcs = { ActivFunc::SIGMOID, ActivFunc::SIGMOID, ActivFunc::SIGMOID };
	model m(layout, funcs, 123, printloss, "../../../../examples/digit_recognition/res/logs");

	dataset<784, 10> testdata;

	if (!ImportMnist(10000, testdata, "../../../../examples/digit_recognition/res/mnist_test"))
		return 1;

	if (load)
	{
		std::ifstream file("../../../../examples/digit_recognition/saved.bin", std::ios::binary);

		model_parameters para;
		para.load_parameters(file);

		model imported(para);
		imported.load(file);

		imported.printo(testdata, 0);
	}
	else
	{
		// !!! You should download the full mnist-set !!!
		// 1. Download mnist_train.csv (from for example https://yann.lecun.com)
		// 2. Copy the header(first line) from mnist_test.csv
		// 3. Put mnist_train.csv in the "res/" folder

		const std::string traindata_path("../../../../examples/digit_recognition/res/mnist_train");
		std::ifstream exists(traindata_path + ".csv");
		if (exists.good() && false)
		{
			dataset<784, 10> traindata;
			if (!ImportMnist(60000, traindata, traindata_path))
				return 1;

			printf("\n");
			m.printo(traindata, 0);
			m.fit(traindata, testdata, runCount, batch_size, thread_count, learning_rate, momentum);
			m.printo(traindata, 0);
		}
		else
		{
			printf("\n");
			m.printo(testdata, 0);
			m.fit(testdata, testdata, runCount, batch_size, thread_count, learning_rate, momentum);
			m.printo(testdata, 0);
		}

		std::ofstream file("../../../../examples/digit_recognition/saved.bin", std::ios::binary);
		m.save(file);
	}

	return 0;
}