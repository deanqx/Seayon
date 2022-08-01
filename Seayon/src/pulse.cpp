#include "seayon.h"

void seayon::pulse()
{
	if (Activation == ActivFunc::SIGMOID)
	{
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

				Layers[l2].Neurons[n2] = Sigmoid(z);
			}
		}
	}
	else if (Activation == ActivFunc::RELU)
	{
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

				Layers[l2].Neurons[n2] = ReLu(z);
			}
		}
	}
	else if (Activation == ActivFunc::REMOID)
	{
		const size_t layer_t = Layers.size() - 2;
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

				Layers[l2].Neurons[n2] = ReLu(z);
			}
		}

		const size_t l1 = layer_t;
		const size_t l2 = layer_t + 1;

		const size_t n2_t = Layers[l2].Neurons.size();
		for (size_t n2 = 0; n2 < n2_t; ++n2)
		{
			float z = 0;
			const size_t n1_t = Layers[l1].Neurons.size();
			for (size_t n1 = 0; n1 < n1_t; ++n1)
				z += Layers[l2].Weights[n2][n1] * Layers[l1].Neurons[n1];
			z += Layers[l2].Biases[n2];

			Layers[l2].Neurons[n2] = Sigmoid(z);
		}
	}
}
void seayon::pulse(std::vector<float>& inputs)
{
	for (size_t n = 0; n < inputs.size(); ++n)
		Layers[0].Neurons[n] = inputs[n];

	pulse();
}

float seayon::cost(std::vector<float>& outputs)
{
	pulse();

	const size_t lLast = Layers.size() - 1;

	float c = 0;
	for (size_t n = 0; n < outputs.size(); ++n)
	{
		float x = Layers[lLast].Neurons[n] - outputs[n];
		c += x * x;
	}

	return c / (float)outputs.size();
}
float seayon::cost(std::vector<float>& inputs, std::vector<float>& outputs)
{
	pulse(inputs);

	const size_t lLast = Layers.size() - 1;

	float x = 0;
	float c = 0;
	for (size_t n = 0; n < outputs.size(); ++n)
	{
		x = Layers[lLast].Neurons[n] - outputs[n];
		c += x * x;
	}

	return c / (float)outputs.size();
}
float seayon::cost(std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& outputs)
{
	const size_t lLast = Layers.size() - 1;

	if (inputs.size() != outputs.size() ||
		inputs[0].size() != Layers[0].Neurons.size() ||
		outputs[0].size() != Layers[lLast].Neurons.size())
	{
		printf("\tCurrupt training data!\n");
		return .0f;
	}

	float acc = 0;
	for (size_t sample = 0; sample < outputs.size(); ++sample)
	{
		acc += cost(inputs[sample], outputs[sample]);
	}

	return acc / (float)outputs.size();
}
float seayon::cost(std::vector<std::vector<std::vector<float>>>& inputs, std::vector<std::vector<std::vector<float>>>& outputs)
{
	const size_t lLast = Layers.size() - 1;

	if (inputs.size() != outputs.size() ||
		inputs[0].size() != Layers[0].Neurons.size() ||
		outputs[0].size() != Layers[lLast].Neurons.size())
	{
		printf("\tCurrupt training data!\n");
		return .0f;
	}

	float acc = 0;
	for (size_t batch = 0; batch < outputs.size(); ++batch)
		for (size_t sample = 0; sample < outputs[batch].size(); ++sample)
		{
			acc += cost(inputs[batch][sample], outputs[batch][sample]);
		}

	return acc / (float)outputs.size() / (float)outputs[0].size();
}

float seayon::accruacy(std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& outputs)
{
	const size_t lLast = Layers.size() - 1;
	const size_t no_t = outputs[0].size();
	const size_t sample_t = inputs.size();

	if (inputs.size() != outputs.size() ||
		inputs[0].size() != Layers[0].Neurons.size() ||
		outputs[0].size() != Layers[lLast].Neurons.size())
	{
		printf("\tCurrupt training data!\n");
		return .0f;
	}

	float a = 0;

	for (size_t i = 0; i < sample_t; ++i)
	{
		pulse(inputs[i]);

		if (std::max_element(Layers[lLast].Neurons.begin(), Layers[lLast].Neurons.end()) - Layers[lLast].Neurons.begin() ==
			std::max_element(outputs[i].begin(), outputs[i].end()) - outputs[i].begin())
		{
			++a;
		}
	}

	return a / (float)sample_t;
}
float seayon::accruacy(std::vector<std::vector<std::vector<float>>>& inputs, std::vector<std::vector<std::vector<float>>>& outputs)
{
	const size_t lLast = Layers.size() - 1;

	if (inputs.size() != outputs.size() ||
		inputs[0].size() != Layers[0].Neurons.size() ||
		outputs[0].size() != Layers[lLast].Neurons.size())
	{
		printf("\tCurrupt training data!\n");
		return .0f;
	}

	float a = 0;
	for (size_t batch = 0; batch < inputs.size(); ++batch)
	{
		a += accruacy(inputs[batch], outputs[batch]);
	}

	return a / (float)inputs.size();
}