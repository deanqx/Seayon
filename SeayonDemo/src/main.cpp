#include "seayon.hpp"

int main()
{
	trainingdata<2, 2, 2> data;
	data.samples[0].inputs[0] = 1.0f;
	data.samples[0].inputs[1] = 0.0f;
	data.samples[0].outputs[0] = 0.0f;
	data.samples[0].outputs[1] = 1.0f;
	data.samples[1].inputs[0] = 0.0f;
	data.samples[1].inputs[1] = 1.0f;
	data.samples[1].outputs[0] = 1.0f;
	data.samples[1].outputs[1] = 0.0f;

	// Input layer size: 2
	// 1. Hidden layer size: 3
	// 2. Hidden layer size: 4
	// Output layer size: 2
	// int layout[]{ 2, 3, 4, 2 };
	int layout[]{ 2, 2 };
	seayon<2> nn(layout, ActivFunc::SIGMOID, 1472); // nn = neural network

	nn.print(); // Prints the whole network to the console

	// ### Before training ###
	nn.printo(data, 0); // Prints only the output layer to the console

	// 50 iterations | 0.5f learning rate | 0.5f momentum
	nn.fit(data, data, 50, true, 0.5f, 0.5f, nullptr, true);

	// ### After training ###
	nn.printo(data, 0);

	return 0;
}