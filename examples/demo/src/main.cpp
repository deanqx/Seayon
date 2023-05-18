#include <vector>
#include "seayon.hpp"

using namespace seayon;

int main()
{
	std::vector<std::vector<float>> inputs = { {1.0f, 0.0f}, { 0.0f, 1.0f} };
	std::vector<std::vector<float>> outputs = { {0.0f, 1.0f}, { 1.0f, 0.0f } };
	dataset data(inputs, outputs);

	// Input layer size: 2
	// 1. Hidden layer size: 3
	// 2. Hidden layer size: 4
	// Output layer size: 2
	// <4 layers> | 2-3-2 neurons | Hidden layer: Leaky ReLu - Output layer: Sigmoid | print current state: true | printing loss after every run: false | seed: 1472 | no logfile
	std::vector<int> layout = { 2, 3, 2 };
	std::vector<ActivFunc> funcs = { ActivFunc::SIGMOID, ActivFunc::SIGMOID };
	model m(layout, funcs, 1472);

	printf("w[0] = np.array([[%f, %f, %f], [%f, %f, %f]])\n",
		m.layers[1].weights[0],
		m.layers[1].weights[2],
		m.layers[1].weights[4],
		m.layers[1].weights[1],
		m.layers[1].weights[3],
		m.layers[1].weights[5]);
	printf("w[2] = np.array([[%f, %f], [%f, %f], [%f, %f]])\n",
		m.layers[2].weights[0 * 3 + 0],
		m.layers[2].weights[1 * 3 + 0],
		m.layers[2].weights[0 * 3 + 1],
		m.layers[2].weights[1 * 3 + 1],
		m.layers[2].weights[0 * 3 + 2],
		m.layers[2].weights[1 * 3 + 2]);

	// ### Before training ###

	// <testdata template> | sample
	// m.print(); // Prints all values to the console
	// testdata | sample (printo: Prints only the output layer to the console)
	// m.printo(data, 1);

	m.print_one(data, 0);
	m.print_one(data, 1);

	// training and test data | Training data shuffle is disabled | 20 epochs
	m.fit(data, data, false, 20, 1, 1, 0.1f);

	// ### After training ###

	// <testdata template> | sample
	// m.pulse<2, 2>(data[0]);
	// m.printo();
	// testdata | sample (printo: Prints only the output layer to the console)
	// m.printo(data, 1);

	printf("%f | %f\n", data[0].x[0], data[0].x[1]);

	m.print_one(data, 0);
	m.print_one(data, 1);

	return 0;
}