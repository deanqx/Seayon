#include <sstream>
#include <iostream>
#include <chrono>
#include "seayon.hpp"

template <int SAMPLES>
bool ImportMnist(trainingdata<SAMPLES, 784, 10>& data, std::ifstream& csv)
{
	auto begin = std::chrono::high_resolution_clock::now();
	auto start = std::chrono::high_resolution_clock::now();

	if (!csv.is_open())
	{
		printf("Cannot open csv file\n");
		return 1;
	}

	std::stringstream buffer;
	buffer << csv.rdbuf();
	std::string file = buffer.str();

	int pos = 0;

	for (; pos < file.size(); ++pos)
		if (file[pos] == '\n')
			break;
	++pos;

	for (int i = 0; i < SAMPLES; ++i)
	{
		auto& sample = data.samples[i];

		std::stringstream label_s;
		for (; pos < file.size() && file[pos] != ','; ++pos)
		{
			label_s << file[pos];
		}
		++pos;

		sample.outputs[stoi(label_s.str())] = 1.0f;

		for (int pixelPos = 0; pixelPos < 784; ++pixelPos)
		{
			std::stringstream pixel;
			for (; file[pos] != ','; ++pos)
			{
				if (file[pos] == '\n')
				{
					sample.inputs[pixelPos] = (float)stoi(pixel.str()) / 255.0f;
					goto Break;
				}

				pixel << file[pos];
			}
			++pos;

			sample.inputs[pixelPos] = (float)stoi(pixel.str()) / 255.0f;
		}

	Break:
		if (i % 500 == 0)
		{
			std::chrono::duration<float> totalelapsed = std::chrono::high_resolution_clock::now() - begin;
			std::chrono::duration<float> elapsed = std::chrono::high_resolution_clock::now() - start;
			start = std::chrono::high_resolution_clock::now();

			int progress = i * 100 / SAMPLES;
			float eta = elapsed.count() * (float)(SAMPLES - i) / 500.0f;
			printf("\t%i%%\tETA: %.0fsec     \tTime: %.0fsec     \t\t\t\r", progress, eta, totalelapsed.count());
		}
	}

	std::chrono::duration<float> totalelapsed = std::chrono::high_resolution_clock::now() - begin;
	printf("\t100%%\t\t\tTime: %.0fsec     \n\n", totalelapsed.count());

	return 0;
}
int main()
{
	constexpr bool load = false;
	constexpr bool print = true;
	constexpr bool printcost = true;

	constexpr int runCount = 1;
	constexpr float learningRate = 0.0003f;
	constexpr float momentum = 0.0001f;

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
		if (train.is_open() && false)
		{
			auto& traindata = *new trainingdata<60000, 784, 10>;
			if (ImportMnist(traindata, train))
				return 1;

			nn.printo(traindata, 0);
			nn.fit(runCount, traindata, testdata, Optimizer::MINI_BATCH, learningRate, momentum, 50);
			nn.printo(traindata, 0);

			delete& traindata;
		}
		else
		{
			nn.printo(testdata, 0);
			nn.fit(runCount, testdata, testdata, Optimizer::MINI_BATCH, learningRate, momentum, 50);
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