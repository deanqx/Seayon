#include <vector>
#include "seayon.hpp"

using namespace seayon;

int main()
{
	dataset<2, 2> data = {
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

	printf("loss: %f\n", m.loss(data));
	float* out = m.pulse<2, 2>(data[0]);
	printf("%f, %f\n", out[0], out[1]);
	out = m.pulse<2, 2>(data[1]);
	printf("%f, %f\n", out[0], out[1]);

	// 20 iterations | training and test data | Stochastic Gradient Descent | 0.5f learning rate | 0.5f momentum
	m.fit(1, data, data, Optimizer::STOCHASTIC, 0.5f, 0.0f);

	// ### After training ###

	// <testdata template> | sample
	// m.pulse<2, 2>(data[0]);
	// m.printo();
	// testdata | sample (printo: Prints only the output layer to the console)
	// m.printo(data, 1);

	printf("loss: %f\n", m.loss(data));
	out = m.pulse<2, 2>(data[0]);
	printf("%f, %f\n", out[0], out[1]);
	out = m.pulse<2, 2>(data[1]);
	printf("%f, %f\n", out[0], out[1]);

	{
		std::vector<char> buffer;
		m.save(buffer);

		model_parameters para;
		para.load_parameters(buffer.data());
		model m2(para);
		m2.load(buffer.data());

		printf("equals: %i seed: %i\n", m.equals(m2), para.seed);
	}
	printf("out of scope\n");

	return 0;
}