#include "seayon.h"

#include <iostream>
#include <iomanip>
#include <chrono>

bool seayon::trainingdata::check(seayon& s)
{
	return s.Layers[0].Neurons.size() == samples[0].inputs.size()
		&& s.Layers[s.Layers.size() - 1].Neurons.size() == samples[0].outputs.size();
}

#define randf(MIN, MAX) MIN + (float)rand() / (float)(RAND_MAX / (MAX - MIN))
void seayon::generate(const std::vector<int> layerCounts, const ActivFunc a, const int seed)
{
	Activation = a;

	if (seed < 0)
		srand(seed);

	std::vector<Layer> flashVector;
	Layers.swap(flashVector);

	Layers.resize(layerCounts.size());
	Layers[0].Neurons.resize(layerCounts[0]);

	for (size_t l2 = 1; l2 < layerCounts.size(); ++l2)
	{
		const size_t l1 = l2 - 1;

		Layers[l2].Neurons.resize(layerCounts[l2]);
		Layers[l2].Biases.resize(layerCounts[l2]);

		if (l2 < layerCounts.size())
		{
			Layers[l2].Weights.resize(layerCounts[l2]);
			for (size_t n2 = 0; n2 < layerCounts[l2]; ++n2)
			{
				Layers[l2].Weights[n2].resize(layerCounts[l1]);
				for (size_t n1 = 0; n1 < layerCounts[l1]; ++n1)
				{
					Layers[l2].Weights[n2][n1] = randf(-2.0f, 2.0f);
				}
			}
		}
	}
}

void resolveTime(int seconds, int* resolved)
{
	resolved[0] = seconds / 3600;
	seconds -= 3600 * resolved[0];
	resolved[1] = seconds / 60;
	seconds -= 60 * resolved[1];
	resolved[2] = seconds;
}

void log(std::ofstream* file, int run, const int runCount, const int sampleCount, int runtime, float elapsed)
{
	float progress = (float)run * 100.0f / (float)runCount;

	int runtimeResolved[3];
	resolveTime(runtime, runtimeResolved);

	int eta[3];
	resolveTime((int)(elapsed * (float)(runCount - run)), eta);

	float samplesPerSecond = (float)sampleCount / (float)elapsed;

	std::ostringstream message;
	message << "\r" << std::setw(4) << "Progress: " << std::fixed << std::setprecision(1) << progress << "% " << std::setw(9)
		<< (int)samplesPerSecond << " Samples/s " << std::setw(13)
		<< "Runtime: " << runtimeResolved[0] << "hours " << runtimeResolved[1] << "min " << runtimeResolved[2] << "sec " << std::setw(9)
		<< "ETA: " << eta[0] << "hours " << eta[1] << "min " << eta[2] << "sec                ";

	std::cout << message.str() << std::flush;
	if (file)
		*file << message.str() << std::endl;
}

template <typename F>
void backpropagate(seayon& s, F activation, F derivative, seayon::trainingdata data, const int runCount, bool print, std::ofstream* file, float N, float M)
{
	auto overall = std::chrono::high_resolution_clock::now();
	auto last = std::chrono::high_resolution_clock::now();
	if (print)
		printf("\n");

	const size_t lLast = s.Layers.size() - 1;

	const size_t sampleCount = data.samples.size();
	std::vector<size_t> Neurons_t(s.Layers.size());
	const size_t l1_t = s.Layers.size();
	for (size_t l1 = 0; l1 < l1_t; ++l1)
	{
		Neurons_t[l1] = s.Layers[l1].Neurons.size();
	}

	std::vector<std::vector<float>> dn(s.Layers.size());
	std::vector<std::vector<std::vector<float>>> lastw(s.Layers.size());
	std::vector<std::vector<float>> lastb(s.Layers.size());
	for (size_t l2 = 1; l2 < s.Layers.size(); ++l2)
	{
		dn[l2].resize(s.Layers[l2].Neurons.size());
		lastw[l2].resize(s.Layers[l2].Neurons.size());
		lastb[l2].resize(s.Layers[l2].Biases.size());
		for (size_t n2 = 0; n2 < s.Layers[l2].Neurons.size(); ++n2)
			lastw[l2][n2].resize(s.Layers[l2].Weights[n2].size());
	}

	if (print)
	{
		log(file, 0, runCount, sampleCount, 0, 0);
	}

	for (int run = 1; run <= runCount; ++run)
	{
		for (size_t sample = 0; sample < sampleCount; ++sample)
		{
			for (size_t n = 0; n < data.samples[sample].inputs.size(); ++n)
				s.Layers[0].Neurons[n] = data.samples[sample].inputs[n];

			const size_t layer_t = s.Layers.size() - 1;
			for (size_t l1 = 0; l1 < layer_t; ++l1)
			{
				const size_t l2 = l1 + 1;

				const size_t n2_t = s.Layers[l2].Neurons.size();
				for (size_t n2 = 0; n2 < n2_t; ++n2)
				{
					float z = 0;
					const size_t n1_t = s.Layers[l1].Neurons.size();
					for (size_t n1 = 0; n1 < n1_t; ++n1)
						z += s.Layers[l2].Weights[n2][n1] * s.Layers[l1].Neurons[n1];
					z += s.Layers[l2].Biases[n2];

					s.Layers[l2].Neurons[n2] = activation(z);
				}
			}

			for (size_t n2 = 0; n2 < Neurons_t[lLast]; ++n2)
			{
				dn[lLast][n2] = derivative(s.Layers[lLast].Neurons[n2]) * 2 * (s.Layers[lLast].Neurons[n2] - data.samples[sample].outputs[n2]);
			}

			size_t l1;
			for (size_t l2 = lLast; l2 >= 2; --l2)
			{
				l1 = l2 - 1;

				for (size_t n1 = 0; n1 < Neurons_t[l1]; ++n1)
				{
					float error = 0;
					for (size_t n2 = 0; n2 < Neurons_t[l2]; ++n2)
						error += dn[l2][n2] * s.Layers[l2].Weights[n2][n1];

					dn[l1][n1] = derivative(s.Layers[l1].Neurons[n1]) * error;
				}
			}

			for (size_t l2 = lLast; l2 >= 1; --l2)
			{
				l1 = l2 - 1;

				for (size_t n2 = 0; n2 < Neurons_t[l2]; ++n2)
				{
					const float db = -dn[l2][n2];
					s.Layers[l2].Biases[n2] += N * db + M * lastb[l2][n2];
					lastb[l2][n2] = db;

					for (size_t n1 = 0; n1 < Neurons_t[l1]; ++n1)
					{
						const float dw = s.Layers[l1].Neurons[n1] * -dn[l2][n2];
						s.Layers[l2].Weights[n2][n1] += N * dw + M * lastw[l2][n2][n1];
						lastw[l2][n2][n1] = dw;
					}
				}
			}
		}
		if (print)
		{
			std::chrono::duration<float> runtime = std::chrono::high_resolution_clock::now() - overall;
			std::chrono::duration<float> elapsed = std::chrono::high_resolution_clock::now() - last;
			last = std::chrono::high_resolution_clock::now();

			log(file, run, runCount, sampleCount, runtime.count(), elapsed.count());
		}
	}

	if (print)
	{
		std::cout << std::endl << std::endl;
	}
}
void seayon::fit(trainingdata& data, const int runCount, const bool print, std::ofstream* logfile, float N, float M)
{
	if (!data.check(*this))
	{
		if (print)
			printf("\tCurrupt training data!\n");
		return;
	}

	if (Activation == ActivFunc::SIGMOID)
	{
		backpropagate(*this, Sigmoid, dSigmoid, data, runCount, print, logfile, N, M);
	}
	else if (Activation == ActivFunc::RELU)
	{
		backpropagate(*this, ReLu, dReLu, data, runCount, print, logfile, N, M);
	}
}