#include <vector>
#include "seayon.h"

int main()
{
	std::vector<std::vector<float>> inputs = {
		{1.0f, 0.0f},
		{0.0f, 1.0f}}; // Two sets of "Input layer values"
	std::vector<std::vector<float>> outputs = {
		{0.0f, 1.0f},
		{1.0f, 0.0f}}; // Two sets of "Input layer values"

	seayon *nn = new seayon; // nn = neural network

	// Input layer size: 2
	// 1. Hidden layer size: 3
	// 2. Hidden layer size: 4
	// Output layer size: 2
	nn->generate(std::vector<int>{2, 3, 4, 2}, seayon::ActivFunc::SIGMOID, 1472); // Randomization seed: 1472

	nn->pulse(inputs[0]); // Calculates the network with first input set
	nn->print();  // Prints the whole network to the console

	// ### Before training ###
	nn->pulse(inputs[0]);
	nn->printo();
	nn->pulse(inputs[1]);
	nn->printo(inputs, outputs); // Prints only the output layer to the console


	// 50 iterations | 0.5f learning rate | 0.5f momentum
	nn->fit(inputs, outputs, 50, 0.5f, 0.5f);


	// ### After training ###
	nn->pulse(inputs[0]);
	nn->printo();
	nn->pulse(inputs[1]);
	nn->printo(inputs, outputs);

	return 0;
}