#include <vector>
#include "seayon.h"

int main()
{
	seayon::trainingdata data = { std::vector<seayon::trainingdata::sample>{
		{std::vector<float>{1.0f, 0.0f}, std::vector<float>{0.0f, 1.0f}},
		{std::vector<float>{0.0f, 1.0f}, std::vector<float>{1.0f, 0.0f}}} }; // Samples[x]: {inputs, outputs}

	seayon* nn = new seayon; // nn = neural network

	// Input layer size: 2
	// 1. Hidden layer size: 3
	// 2. Hidden layer size: 4
	// Output layer size: 2
	nn->generate(std::vector<int>{2, 3, 4, 2}, seayon::ActivFunc::SIGMOID, 1472); // Randomization seed: 1472

	nn->print(); // Prints the whole network to the console

	// ### Before training ###
	nn->printo(data, 0); // Prints only the output layer to the console

	// 50 iterations | 0.5f learning rate | 0.5f momentum
	nn->fit(data, 50, false, nullptr, 0.5f, 0.5f);

	// ### After training ###
	nn->printo(data, 0);

	return 0;
}