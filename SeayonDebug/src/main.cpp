#include <vector>
#include "Seayon.h"

int main()
{
	std::vector<std::vector<std::vector<float>>> inputs = {
		{
			{ 1, 0 },
			{ 0, 1 }
		}
	};
	std::vector<std::vector<std::vector<float>>> outputs = {
		{
			{ 0, 1 },
			{ 1, 0 }
		}
	};

	seayon* nn = new seayon;
	nn->generate(std::vector<unsigned>{ 2, 64, 64, 2 }, seayon::ActivFunc::SIGMOID, 1472);
	nn->generate(std::vector<unsigned>{ 2, 2, 2 }, seayon::ActivFunc::RELU, 1472);

	nn->pulse(inputs[0][0]);
	nn->print();
	nn->pulse(inputs[0][1]);
	nn->printo(inputs[0], outputs[0]);

	nn->fit(inputs[0], outputs[0], 25, 0.03, 0.1);

	nn->pulse(inputs[0][0]);
	nn->print();
	nn->pulse(inputs[0][1]);
	nn->printo(inputs[0], outputs[0]);

	system("pause");
}