#include <iostream>
#include <vector>
#include <chrono>
#include <sstream>
#include <fstream>
#include <streambuf>
#include "Seayon.h"

void ImportHistory(std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& outputs, unsigned samples, std::string csvPath)
{
	inputs.resize(samples);
	outputs.resize(samples);
	std::ifstream file(csvPath);

	std::string line;
	size_t lseek;

	std::getline(file, line);
	for (unsigned sample = 0; sample < samples && std::getline(file, line); ++sample)
	{
		auto start = std::chrono::high_resolution_clock::now();

		lseek = 0;
		std::vector<float> _inputs(800);
		std::vector<float> _outputs(2);

		for (size_t out = 0; out < 2; ++out)
		{
			std::stringstream number;
			for (; line[lseek] != ','; ++lseek)
				number << line[lseek];
			++lseek;

			_outputs[out] = std::stof(number.str());
		}

		for (size_t in = 0; in < 800; ++in)
		{
			std::stringstream number;
			for (; line[lseek] != ','; ++lseek)
				number << line[lseek];
			++lseek;

			_inputs[in] = std::stof(number.str());
		}

		inputs[sample] = _inputs;
		outputs[sample] = _outputs;

		if (sample % 100 == 0)
		{
			std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;
			float eta = (float)elapsed.count() * (float)(samples - sample) / 60.0f;
			printf("\rTime left: %.3fmin", eta);
		}
	}

	file.close();
}

int main()
{
	std::vector<std::vector<float>> inputs;
	std::vector<std::vector<float>> outputs;

	seayon* nn = new seayon;

	nn->generate(std::vector<unsigned>{ 800, 256, 256, 2 }, seayon::ActivFunc::SIGMOID, 1472);

	ImportHistory(inputs, outputs, 10000, "C:\\Users\\dean\\AppData\\Roaming\\MetaQuotes\\Tester\\FB965BE7DC4B16DDF7FC73CCD9453B43\\Agent-127.0.0.1-3000\\MQL5\\Files\\history.csv");

	nn->pulse(inputs[0]);
	nn->printo();

	printf("Before Acc: %.2f\n", nn->cost(inputs, outputs));
	
	nn->fit(inputs, outputs, 1);

	printf("After Acc:  %.2f\n", nn->cost(inputs, outputs));

	printf("Second test");

	std::ofstream file("C:\\Users\\dean\\AppData\\Roaming\\MetaQuotes\\Tester\\FB965BE7DC4B16DDF7FC73CCD9453B43\\Agent-127.0.0.1-3000\\MQL5\\Files\\test.nn");
	nn->save(file, false);
	file.close();

	//nn->pulse(inputs[0][1]);
	//nn->printo();

	//printf("Cost: %f", nn->Cost(inputs, outputs));
	//nn->pulse(inputs[0]);
	//nn->printo();

	delete nn;

	system("pause");
	return 0;
}