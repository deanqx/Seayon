#include <vector>
#include "seayon.hpp"

int main()
{
	trainingdata<2, 2> data = {
		{ {1.0f, 0.0f}, { 0.0f, 1.0f }},
		{ {0.0f, 1.0f}, {1.0f, 0.0f} }
	};

	// Input layer size: 2
	// 1. Hidden layer size: 3
	// 2. Hidden layer size: 4
	// Output layer size: 2
	// <4 layers> | 2-3-4-2 neurons | Hidden layer: Leaky ReLu - Output layer: Sigmoid | print current state: true | printing cost after every run: false | seed: 1472 | no logfile
	std::vector<int> layout = { 2, 3, 4, 2 };
	std::vector<ActivFunc> funcs = { ActivFunc::RELU, ActivFunc::RELU, ActivFunc::RELU, ActivFunc::SIGMOID };
	seayon nn(layout, funcs, false, 1472, "");

	// ### Before training ###

	// <testdata template> | sample
	nn.pulse<2, 2>(data[0]);
	nn.print(); // Prints all values to the console
	// testdata | sample (printo: Prints only the output layer to the console)
	nn.printo(data, 1);

	// 20 iterations | training and test data | Stochastic Gradient Descent | 0.5f learning rate | 0.5f momentum
	nn.fit(20, data, data, Optimizer::STOCHASTIC, 0.5f, 0.5f);

	// ### After training ###

	// <testdata template> | sample
	nn.pulse<2, 2>(data[0]);
	nn.printo();
	// testdata | sample (printo: Prints only the output layer to the console)
	nn.printo(data, 1);

	return 0;
}