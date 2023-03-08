#include "seayon.h"

#include <iostream>
#include <chrono>

template <typename F>
inline void seayon::backpropagate(F activation, F derivative, std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& outputs, int& runs, bool& print, float& N, float& M)
{
	auto start = std::chrono::high_resolution_clock::now();
	auto last = std::chrono::high_resolution_clock::now();
	if (print)
		printf("\n");

	const size_t lLast = Layers.size() - 1;

	const size_t sample_t = inputs.size();
	std::vector<size_t> Neurons_t(Layers.size());
	const size_t l1_t = Layers.size();
	for (size_t l1 = 0; l1 < l1_t; ++l1)
	{
		Neurons_t[l1] = Layers[l1].Neurons.size();
	}

	std::vector<std::vector<float>> dn(Layers.size());
	std::vector<std::vector<std::vector<float>>> lastw(Layers.size());
	std::vector<std::vector<float>> lastb(Layers.size());
	for (size_t l2 = 1; l2 < Layers.size(); ++l2)
	{
		dn[l2].resize(Layers[l2].Neurons.size());
		lastw[l2].resize(Layers[l2].Neurons.size());
		lastb[l2].resize(Layers[l2].Biases.size());
		for (size_t n2 = 0; n2 < Layers[l2].Neurons.size(); ++n2)
			lastw[l2][n2].resize(Layers[l2].Weights[n2].size());
	}

	float spsTotal = 0.0f;

	if (print)
	{
		printf("\r\tTraining 0.0%%      Runtime: 0hours 0min 0sec          0.0 Samples/s \tETA: N/A                          ");
	}

	for (int run = 1; run <= runs; ++run)
	{
		for (size_t sample = 0; sample < sample_t; ++sample)
		{
			for (size_t n = 0; n < inputs[sample].size(); ++n)
				Layers[0].Neurons[n] = inputs[sample][n];

			const size_t layer_t = Layers.size() - 1;
			for (size_t l1 = 0; l1 < layer_t; ++l1)
			{
				const size_t l2 = l1 + 1;

				const size_t n2_t = Layers[l2].Neurons.size();
				for (size_t n2 = 0; n2 < n2_t; ++n2)
				{
					float z = 0;
					const size_t n1_t = Layers[l1].Neurons.size();
					for (size_t n1 = 0; n1 < n1_t; ++n1)
						z += Layers[l2].Weights[n2][n1] * Layers[l1].Neurons[n1];
					z += Layers[l2].Biases[n2];

					Layers[l2].Neurons[n2] = activation(z);
				}
			}

			for (size_t n2 = 0; n2 < Neurons_t[lLast]; ++n2)
			{
				dn[lLast][n2] = derivative(Layers[lLast].Neurons[n2]) * 2 * (Layers[lLast].Neurons[n2] - outputs[sample][n2]);
			}

			size_t l1;
			for (size_t l2 = lLast; l2 >= 2; --l2)
			{
				l1 = l2 - 1;

				for (size_t n1 = 0; n1 < Neurons_t[l1]; ++n1)
				{
					float error = 0;
					for (size_t n2 = 0; n2 < Neurons_t[l2]; ++n2)
						error += dn[l2][n2] * Layers[l2].Weights[n2][n1];

					dn[l1][n1] = derivative(Layers[l1].Neurons[n1]) * error;
				}
			}

			for (size_t l2 = lLast; l2 >= 1; --l2)
			{
				l1 = l2 - 1;

				for (size_t n2 = 0; n2 < Neurons_t[l2]; ++n2)
				{
					const float db = -dn[l2][n2];
					Layers[l2].Biases[n2] += N * db + M * lastb[l2][n2];
					lastb[l2][n2] = db;

					for (size_t n1 = 0; n1 < Neurons_t[l1]; ++n1)
					{
						const float dw = Layers[l1].Neurons[n1] * -dn[l2][n2];
						Layers[l2].Weights[n2][n1] += N * dw + M * lastw[l2][n2][n1];
						lastw[l2][n2][n1] = dw;
					}
				}
			}
		}
		if (print)
		{
			std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;
			int time0 = (int)elapsed.count();
			const int h0 = time0 / 3600;
			time0 -= 3600 * h0;
			const int m0 = time0 / 60;
			time0 -= 60 * m0;
			const int s0 = time0;

			std::chrono::duration<double> runElapsed = std::chrono::high_resolution_clock::now() - last;
			last = std::chrono::high_resolution_clock::now();

			int time1 = (int)runElapsed.count() * (runs - run);
			const int h1 = time1 / 3600;
			time1 -= 3600 * h1;
			const int m1 = time1 / 60;
			time1 -= 60 * m1;
			const int s1 = time1;

			const float sps = (float)sample_t / (float)runElapsed.count();
			spsTotal += sps;

			printf("\r\tTraining %.1f%%      Runtime: %ihours %imin %isec          %.0f Samples/s \tETA: %ihours %imin %isec            ", (float)run * 100.0f / (float)runs, h0, m0, s0, sps, h1, m1, s1);
		}
	}

	if (print)
	{
		std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;
		int time0 = (int)elapsed.count();
		const int h0 = time0 / 3600;
		time0 -= 3600 * h0;
		const int m0 = time0 / 60;
		time0 -= 60 * m0;
		const int s0 = time0;

		printf("\r\tTraining 100.0%%      Runtime: %ihours %imin %isec     avg. %.0f Samples/s                                         \n\n", h0, m0, s0, spsTotal / runs);
	}
}
void seayon::fit(std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& outputs, int runs, bool print, float N, float M)
{
	if (inputs.size() != outputs.size() ||
		inputs[0].size() != Layers[0].Neurons.size() ||
		outputs[0].size() != Layers[Layers.size() - 1].Neurons.size())
	{
		if (print)
			printf("\tCurrupt training data!\n");
		return;
	}

	if (Activation == ActivFunc::SIGMOID)
	{
		backpropagate(Sigmoid, dSigmoid, inputs, outputs, runs, print, N, M);
	}
	else if (Activation == ActivFunc::RELU)
	{
		backpropagate(ReLu, dReLu, inputs, outputs, runs, print, N, M);
	}
}