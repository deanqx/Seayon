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
	// <4 layers> | 2-3-2 neurons | Hidden layer: Leaky ReLu - Output layer: Sigmoid | print current state: true | printing loss after every run: false | seed: 1472 | no logfile
	std::vector<int> layout = { 2, 3, 2 };
	std::vector<ActivFunc> funcs = { ActivFunc::SIGMOID, ActivFunc::SIGMOID };
	seayon nn(layout, funcs, 1472);

	printf("w[0] = np.array([[%f, %f, %f], [%f, %f, %f]])\n",
		nn.layers[1].weights[0],
		nn.layers[1].weights[2],
		nn.layers[1].weights[4],
		nn.layers[1].weights[1],
		nn.layers[1].weights[3],
		nn.layers[1].weights[5]);
	printf("w[2] = np.array([[%f, %f], [%f, %f], [%f, %f]])\n",
		nn.layers[2].weights[0 * 3 + 0],
		nn.layers[2].weights[1 * 3 + 0],
		nn.layers[2].weights[0 * 3 + 1],
		nn.layers[2].weights[1 * 3 + 1],
		nn.layers[2].weights[0 * 3 + 2],
		nn.layers[2].weights[1 * 3 + 2]);

	// ### Before training ###

	// <testdata template> | sample
	// nn.print(); // Prints all values to the console
	// testdata | sample (printo: Prints only the output layer to the console)
	// nn.printo(data, 1);

	printf("loss: %f\n", nn.loss(data));
	float* out = nn.pulse<2, 2>(data[0]);
	printf("%f, %f\n", out[0], out[1]);
	out = nn.pulse<2, 2>(data[1]);
	printf("%f, %f\n", out[0], out[1]);

	// 20 iterations | training and test data | Stochastic Gradient Descent | 0.5f learning rate | 0.5f momentum
	nn.fit(1, data, data, Optimizer::STOCHASTIC, 0.5f, 0.0f);

	// ### After training ###

	// <testdata template> | sample
	// nn.pulse<2, 2>(data[0]);
	// nn.printo();
	// testdata | sample (printo: Prints only the output layer to the console)
	// nn.printo(data, 1);

	printf("loss: %f\n", nn.loss(data));
	out = nn.pulse<2, 2>(data[0]);
	printf("%f, %f\n", out[0], out[1]);
	out = nn.pulse<2, 2>(data[1]);
	printf("%f, %f\n", out[0], out[1]);

	return 0;
}