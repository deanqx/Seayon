#include "seayon.h"

#include <iostream>
#include <algorithm>

template <typename F>
inline void _pulse(seayon& s, F func)
{
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

			s.Layers[l2].Neurons[n2] = func(z);
		}
	}
}

void seayon::pulse(trainingdata::sample& sample)
{
	for (size_t n = 0; n < sample.inputs.size(); ++n)
		Layers[0].Neurons[n] = sample.inputs[n];

	if (Activation == ActivFunc::SIGMOID)
	{
		_pulse(*this, Sigmoid);
	}
	else if (Activation == ActivFunc::RELU)
	{
		_pulse(*this, ReLu);
	}
}

float seayon::cost(trainingdata::sample& sample)
{
	pulse(sample);

	const size_t lLast = Layers.size() - 1;

	float c = 0;
	for (size_t n = 0; n < sample.outputs.size(); ++n)
	{
		float x = Layers[lLast].Neurons[n] - sample.outputs[n];
		c += x * x;
	}

	return c / (float)sample.outputs.size();
}
float seayon::cost(trainingdata& data)
{
	const size_t lLast = Layers.size() - 1;

	if (!data.check(*this))
	{
		printf("\tCurrupt training data!\n");
		return .0f;
	}

	float acc = 0;
	for (size_t i = 0; i < data.samples.size(); ++i)
	{
		acc += cost(data.samples[i]);
	}

	return acc / (float)data.samples.size();
}

float seayon::accruacy(trainingdata& data)
{
	const size_t lLast = Layers.size() - 1;
	const size_t no_t = data.samples[0].outputs.size();

	if (!data.check(*this))
	{
		printf("\tCurrupt training data!\n");
		return .0f;
	}

	float a = 0;
	for (size_t i = 0; i < data.samples.size(); ++i)
	{
		pulse(data.samples[i]);

		if (std::max_element(Layers[lLast].Neurons.begin(), Layers[lLast].Neurons.end()) - Layers[lLast].Neurons.begin() ==
			std::max_element(data.samples[i].outputs.begin(), data.samples[i].outputs.end()) - data.samples[i].outputs.begin())
		{
			++a;
		}
	}

	return a / (float)data.samples.size();
}