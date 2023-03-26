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
	// <4 layers> | 2-3-4-2 neurons | Activation function: Sigmoid | print current state: true | printing cost after every run: false | seed: 1472 | no logfile
	int layout[]{2, 3, 4, 2};
	seayon<4> nn(layout, ActivFunc::SIGMOID, true, false, 1472, "");

	nn.print(); // Prints all values to the console

	// ### Before training ###

	// <testdata template> | sample
	nn.pulse<2, 2, 2>(data.samples[0]);
	nn.printo();
	// testdata | sample (printo: Prints only the output layer to the console)
	nn.printo(data, 1);

	// 50 iterations | training and test data | 0.5f learning rate | 0.5f momentum
	nn.fit(50, data, data, 0.5f, 0.5f);

	// ### After training ###

	// <testdata template> | sample
	nn.pulse<2, 2, 2>(data.samples[0]);
	nn.printo();
	// testdata | sample (printo: Prints only the output layer to the console)
	nn.printo(data, 1);

	return 0;
}