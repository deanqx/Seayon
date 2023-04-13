#include <sstream>
#include <iostream>
#include <chrono>
#include "seayon.hpp"

template <int SAMPLES>
static void parse_line(const std::string* lines, const int begin, const int end, trainingdata<SAMPLES, 784, 10>& data)
{
	for (int i = begin; i <= end; ++i)
	{
		int pos = 0;
		auto& sample = data.samples[i];

		std::stringstream label;
		for (; lines[i][pos] != ','; ++pos)
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
}

template <int SAMPLES>
bool ImportMnist(trainingdata<SAMPLES, 784, 10>& data, std::ifstream& csv, int thread_count = 32)
{
	const int per_thread = SAMPLES / thread_count;

	if (!csv.is_open())
	{
		printf("Cannot open csv file\n");
		return 1;
	}

	printf("\tLoading mnist...");

	std::string lines[SAMPLES];
	std::vector<std::thread> threads(thread_count);

	std::getline(csv, lines[0]); // garbage
	for (int i = 0; i < SAMPLES; ++i)
	{
		std::getline(csv, lines[i]);
	}

	for (int t = 0; t < thread_count; ++t)
	{
		threads[t] = std::thread([&, t]
			{
				const int begin = t * per_thread;
				const int end = begin + per_thread - 1;
				parse_line(lines, begin, end, data);
			});
	}

	const int begin = thread_count * per_thread;
	const int end = SAMPLES - 1;
	parse_line(lines, begin, end, data);

	for (int t = 0; t < thread_count; ++t)
	{
		threads[t].join();
	}

	printf(" DONE\n");

	return 0;
}
int main()
{
	constexpr bool load = false;
	constexpr bool print = true;
	constexpr bool printcost = true;

	constexpr int runCount = 100;
	constexpr float learningRate = 0.00003f;
	constexpr float momentum = 0.00001f;
	constexpr float batch_size = 100;
	constexpr float thread_count = 32;

	int layout[]{ 784, 16, 16, 10 };
	ActivFunc funcs[]{ ActivFunc::SIGMOID, ActivFunc::SIGMOID, ActivFunc::SIGMOID, ActivFunc::SIGMOID };
	seayon nn(layout, funcs, print, printcost, 1472, "../../../../SeayonMnist/res/logs");

	auto& testdata = *new trainingdata<10000, 784, 10>;

	std::ifstream test("../../../../SeayonMnist/res/mnist/mnist_test.csv");
	if (ImportMnist(testdata, test))
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
			auto& traindata = *new trainingdata<60000, 784, 10>;
			if (ImportMnist(traindata, train))
				return 1;

			printf("\n");
			nn.printo(traindata, 0);
			nn.fit(runCount, traindata, testdata, Optimizer::MINI_BATCH, learningRate, momentum, batch_size, thread_count);
			nn.printo(traindata, 0);

			delete& traindata;
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

	nn.clean();
	delete& testdata;

	return 0;
}