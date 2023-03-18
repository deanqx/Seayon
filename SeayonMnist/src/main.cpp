#include <sstream>
#include <iostream>
#include <chrono>
#include "seayon.hpp"

template<int SAMPLES>
void ImportMnist(trainingdata<SAMPLES, 784, 10>& data, std::ifstream& csv)
{
	auto begin = std::chrono::high_resolution_clock::now();
	auto start = std::chrono::high_resolution_clock::now();

	if (!csv.is_open())
	{
		printf("Cannot open csv file\n");
		return;
	}

	std::string line;
	std::getline(csv, line);

	for (int i = 0; i < SAMPLES; ++i)
	{
		std::getline(csv, line);
		auto& sample = data.samples[i];

		int pos = 0;
		std::stringstream label_s;
		for (; pos < line.size(); ++pos)
		{
			if (line[pos] == ',')
				break;

			label_s << line[pos];
		}
		++pos;

		sample.outputs[stoi(label_s.str())] = 1.0f;

		for (int pixelPos = 0; pixelPos < 784; ++pixelPos)
		{
			std::stringstream pixel;
			for (;; ++pos)
			{
				if (pos >= line.size())
				{
					sample.inputs[pixelPos] = (float)stoi(pixel.str()) / 255.0f;
					goto Break;
				}
				if (line[pos] == ',')
					break;

				pixel << line[pos];
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
}
int main()
{
	const bool load = false;

	int layerNeurons[]{ 784, 16, 16, 10 };
	seayon<4> nn(layerNeurons, ActivFunc::SIGMOID, 1472);

	auto& testdata = *new trainingdata<10000, 784, 10>();

	std::ifstream test("res/mnist/mnist_test.csv");
	ImportMnist(testdata, test);
	test.close();

	if (load)
	{
		std::ifstream file("res/mnist.bin");
		nn.load(file);
		file.close();
	}
	else
	{
		// !!! You should download the full mnist-set !!!
		// 1. Download mnist_train.csv (from for example https://yann.lecun.com)
		// 2. Copy the header(first line) from mnist_test.csv
		// 3. Put mnist_train.csv in the "res/mnist/" folder

		std::ifstream train("res/mnist/mnist_train.csv");
		if (train.is_open())
		{
			auto& traindata = *new trainingdata<60000, 784, 10>();
			ImportMnist(traindata, train);

			nn.printo(traindata, 0);
			nn.fit(traindata, testdata, 50, true, 0.03f, 0.1f, nullptr, false);
			nn.printo(traindata, 0);

			delete& traindata;
		}
		else
		{
			nn.printo(testdata, 0);
			nn.fit(testdata, testdata, 50, true, 0.03f, 0.1f, nullptr, false);
			nn.printo(testdata, 0);
		}
		train.close();

		std::ofstream file("res/mnist.bin");
		nn.save(file);
		file.close();
	}

	nn.printo(testdata, 0);

	nn.clean();
	delete& testdata;

	return 0;
}