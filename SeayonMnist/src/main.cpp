#include <vector>
#include <sstream>
#include <iostream>
#include <chrono>
#include <thread>
#include <fstream>
#include <stdio.h>
#include "seayon.h"

void ImportMnist(std::vector<std::vector<std::vector<float>>>& inputs, std::vector<std::vector<std::vector<float>>>& outputs, unsigned batchSize, unsigned totalLenght, std::string csvPath)
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
		std::cerr << "Cannot open csv file" << std::endl;
		return;
	}

	std::getline(csv, line);

	int batch = -1;
	for (unsigned package = 0; std::getline(csv, line); ++package)
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

		if (package % 100 == 0)
		{
			std::chrono::duration<double> totalelapsed = std::chrono::high_resolution_clock::now() - begin;
			std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;
			start = std::chrono::high_resolution_clock::now();

			unsigned progress = package * 100 / totalLenght;
			float eta = (float)elapsed.count() / 100.f * (float)(totalLenght - package);
			std::cout << "\t" << progress << "%\tETA: " << (unsigned)eta << "sec\t" << (int)totalelapsed.count() << "sec\t\t\t\r";
		}
	}
	std::cout << "\t100%\t\t\t\t\r" << std::endl << std::endl;
}
int main()
{
	bool load = false;


	std::vector<std::vector<std::vector<float>>> inputs;
	std::vector<std::vector<std::vector<float>>> outputs;

	seayon* nn = new seayon;

	nn->generate(std::vector<unsigned>{ 784, 16, 16, 10 }, seayon::ActivFunc::SIGMOID, 1472);

	if (load)
	{
		std::ifstream file("C:\\Stack\\Projects\\Workspace\\Seayon\\SeayonMnist\\res\\test.nn");
		nn->load(file, false);
		file.close();
	}
	else
	{
		ImportMnist(inputs, outputs, 60000, 60000, "C:\\Stack\\Projects\\Workspace\\Seayon\\SeayonMnist\\res\\mnist\\mnist_train.csv");

		nn->pulse(inputs[0][1]);
		nn->printo(inputs[0], outputs[0]);

		nn->fit(inputs[0], outputs[0], 5, true, 0.03f, 0.1f);

		nn->pulse(inputs[0][1]);
		nn->printo(inputs[0], outputs[0]);
	}

	ImportMnist(inputs, outputs, 10000, 10000, "C:\\Stack\\Projects\\Workspace\\Seayon\\SeayonMnist\\res\\mnist\\mnist_test.csv");

	nn->pulse(inputs[0][1]);
	nn->printo(inputs[0], outputs[0]);
	
	if (!load)
	{
		std::ofstream file("C:\\Stack\\Projects\\Workspace\\Seayon\\SeayonMnist\\res\\test.nn");
		nn->save(file, false);
		file.close();
	}

	delete nn;

	return 0;
}