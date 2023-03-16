#include <vector>
#include <sstream>
#include <iostream>
#include <chrono>
#include <thread>
#include <fstream>
#include <stdio.h>
#include "seayon.h"

seayon::trainingdata* ImportMnist(int batchCount, int sampleCount, std::ifstream& csv)
{
	sampleCount /= batchCount;

	seayon::trainingdata* data = new seayon::trainingdata[batchCount]();
	for (int batch = 0; batch < batchCount; ++batch)
	{
		data[batch].samples = std::vector<seayon::trainingdata::sample>(sampleCount);
	}

	auto begin = std::chrono::high_resolution_clock::now();
	auto start = std::chrono::high_resolution_clock::now();

	if (!csv.is_open())
	{
		printf("Cannot open csv file\n");
		return nullptr;
	}

	std::string line;
	std::getline(csv, line);

	for (int batch = 0; batch < batchCount; ++batch)
	{
		for (int i = 0; i < sampleCount; ++i)
		{
			std::getline(csv, line);
			auto& sample = data[batch].samples[i];

			int pos = 0;
			std::stringstream label_s;
			for (; pos < line.size(); ++pos)
			{
				if (line[pos] == ',')
					break;

				label_s << line[pos];
			}
			++pos;

			switch (stoi(label_s.str()))
			{
			case 0:
				sample.outputs = { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
				break;
			case 1:
				sample.outputs = { 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
				break;
			case 2:
				sample.outputs = { 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
				break;
			case 3:
				sample.outputs = { 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
				break;
			case 4:
				sample.outputs = { 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
				break;
			case 5:
				sample.outputs = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f };
				break;
			case 6:
				sample.outputs = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f };
				break;
			case 7:
				sample.outputs = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f };
				break;
			case 8:
				sample.outputs = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f };
				break;
			case 9:
				sample.outputs = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f };
				break;
			}

			const size_t pixelImage_t = 784;
			float pixelImage[pixelImage_t];
			for (int pixelPos = 0; pixelPos < pixelImage_t; ++pixelPos)
			{
				std::stringstream pixel;
				for (;; ++pos)
				{
					if (pos >= line.size())
					{
						pixelImage[pixelPos] = (float)stoi(pixel.str()) / 255.0f;
						goto Break;
					}
					if (line[pos] == ',')
						break;

					pixel << line[pos];
				}
				++pos;

				pixelImage[pixelPos] = (float)stoi(pixel.str()) / 255.0f;
			}
		Break:
			sample.inputs = std::vector<float>(pixelImage, pixelImage + pixelImage_t);

			int total = batch * batchCount + i;
			if (total % 500 == 0)
			{
				std::chrono::duration<float> totalelapsed = std::chrono::high_resolution_clock::now() - begin;
				std::chrono::duration<float> elapsed = std::chrono::high_resolution_clock::now() - start;
				start = std::chrono::high_resolution_clock::now();

				int totalNeeded = batch * batchCount + sampleCount;

				int progress = total * 100 / totalNeeded;
				float eta = elapsed.count() * (float)(totalNeeded - total) / 500.0f;
				printf("\t%i%%\tETA: %.0fsec     \tTime: %.0fsec     \t\t\t\r", progress, eta, totalelapsed.count());
			}
		}
	}

	std::chrono::duration<float> totalelapsed = std::chrono::high_resolution_clock::now() - begin;
	printf("\t100%%\t\t\tTime: %.0fsec     \n\n", totalelapsed.count());

	return data;
}
int main()
{
	const bool load = false;

	seayon::trainingdata* data; // data[0]: First batch
	seayon* nn = new seayon;

	nn->generate(std::vector<int>{784, 16, 16, 10}, seayon::ActivFunc::SIGMOID, 1472);

	if (load)
	{
		std::ifstream file("res/mnist.nn");
		nn->load(file);
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
			data = ImportMnist(1, 60000, train);
		}
		else
		{
			train.close();
			train.clear();
			train.open("res/mnist/mnist_test.csv");
			data = ImportMnist(1, 10000, train);
		}
		train.close();

		nn->printo(data[0], 0);

		nn->fit(data[0], 50, true, nullptr, 0.03f, 0.1f);

		nn->printo(data[0], 0);

		std::ofstream file("res/mnist.nn");
		nn->save(file);
		file.close();
	}

	std::ifstream test("res/mnist/mnist_test.csv");
	data = ImportMnist(1, 10000, test);
	test.close();

	nn->printo(data[0], 0);

	delete nn;

	return 0;
}