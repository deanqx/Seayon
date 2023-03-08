#include <vector>
#include <sstream>
#include <iostream>
#include <chrono>
#include <thread>
#include <fstream>
#include <stdio.h>
#include "seayon.h"

void ImportMnist(std::vector<std::vector<std::vector<float>>>& inputs, std::vector<std::vector<std::vector<float>>>& outputs, int batchSize, int totalLenght, std::string csvPath)
{
	std::vector<std::vector<std::vector<float>>> flash_inputs;
	std::vector<std::vector<std::vector<float>>> flash_outputs;

	inputs.swap(flash_inputs);
	outputs.swap(flash_outputs);

	auto begin = std::chrono::high_resolution_clock::now();
	auto start = std::chrono::high_resolution_clock::now();

	std::ifstream csv(csvPath);
	std::string line;

	if (!csv.is_open())
	{
		printf("Cannot open csv file\n");
		return;
	}

	std::getline(csv, line);

	int batch = -1;
	for (int package = 0; std::getline(csv, line); ++package)
	{
		if (package % batchSize == 0)
		{
			++batch;

			std::vector<std::vector<float>> temp0;
			std::vector<std::vector<float>> temp1;

			inputs.push_back(temp0);
			outputs.push_back(temp1);
		}

		size_t pos = 0;
		std::stringstream label_s;
		for (; pos < line.size(); ++pos)
		{
			if (line[pos] == ',')
				break;

			label_s << line[pos];
		}
		++pos;

		std::vector<float> label(10);
		switch (stoi(label_s.str()))
		{
		case 0:
			label = { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
			break;
		case 1:
			label = { 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
			break;
		case 2:
			label = { 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
			break;
		case 3:
			label = { 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
			break;
		case 4:
			label = { 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
			break;
		case 5:
			label = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f };
			break;
		case 6:
			label = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f };
			break;
		case 7:
			label = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f };
			break;
		case 8:
			label = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f };
			break;
		case 9:
			label = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f };
			break;
		}
		outputs[batch].push_back(label);

		const size_t pixelImage_t = 784;
		float pixelImage[pixelImage_t];
		for (int pixelPos = 0; ; ++pixelPos)
		{
			std::stringstream pixel;
			for (; ; ++pos)
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
		inputs[batch].push_back(std::vector<float>(pixelImage, pixelImage + pixelImage_t));

		if (package % 500 == 0)
		{
			std::chrono::duration<double> totalelapsed = std::chrono::high_resolution_clock::now() - begin;
			std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;
			start = std::chrono::high_resolution_clock::now();

			int progress = package * 100 / totalLenght;
			float eta = (float)elapsed.count() * (float)(totalLenght - package) / 500.0f;
			printf("\t%i%%\tETA: %isec\t%isec\t\t\t\r", progress, (int)eta, (int)totalelapsed.count());
		}
	}
	printf("\t100%\t\t\t\t\n\n");
}
int main()
{
	const bool load = false;

	std::vector<std::vector<std::vector<float>>> inputs;
	std::vector<std::vector<std::vector<float>>> outputs;

	seayon* nn = new seayon;

	nn->generate(std::vector<int>{ 784, 16, 16, 10 }, seayon::ActivFunc::SIGMOID, 1472);

	if (load)
	{
		std::ifstream file("../SeayonMnist/res/mnist.nn");
		nn->load(file, false);
		file.close();
	}
	else
	{
		// !!! You should download the full mnist-set !!!
		// 1. Download mnist_train.csv (from for example https://yann.lecun.com)
		// 2. Copy the header(first line) from mnist_test.csv
		// 3. Put mnist_train.csv in the "../SeayonMnist/res/mnist/" folder
		// 4. Replace with: ImportMnist(inputs, outputs, 60000, 60000, "../SeayonMnist/res/mnist/mnist_train.csv");

		ImportMnist(inputs, outputs, 10000, 10000, "../SeayonMnist/res/mnist/mnist_test.csv");

		nn->pulse(inputs[0][1]);
		nn->printo(inputs[0], outputs[0]);

		nn->fit(inputs[0], outputs[0], 50, true, 0.03f, 0.1f);

		nn->pulse(inputs[0][1]);
		nn->printo(inputs[0], outputs[0]);

		std::ofstream file("../SeayonMnist/res/mnist.nn");
		nn->save(file, false);
		file.close();
	}

	ImportMnist(inputs, outputs, 10000, 10000, "../SeayonMnist/res/mnist/mnist_test.csv");

	nn->pulse(inputs[0][1]);
	nn->printo(inputs[0], outputs[0]);

	delete nn;

	return 0;
}