#include "../seayon.hpp"

float randf(float min, float max)
{
	return min + (float)rand() / (float)(RAND_MAX / (max - min));
}

seayon::model::layer::layer(const int PREVIOUS, const int NEURONS, const ActivFunc func, std::mt19937& gen)
	: nCount(NEURONS), wCount(NEURONS* PREVIOUS), func(func)
{
	neurons.resize(nCount);

	if (wCount > 0)
	{
		biases.resize(nCount);
		weights.reserve(wCount);

		std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
		for (int i = 0; i < wCount; ++i)
		{
			weights.push_back(dis(gen));
		}
	}

	if (func == ActivFunc::SIGMOID)
	{
		activation = Sigmoid;
		derivative = dSigmoid;
	}
	else if (func == ActivFunc::TANH)
	{
		activation = Tanh;
		derivative = dTanh;
	}
	else if (func == ActivFunc::RELU)
	{
		activation = ReLu;
		derivative = dReLu;
	}
	else if (func == ActivFunc::LEAKYRELU)
	{
		activation = LeakyReLu;
		derivative = dLeakyReLu;
	}
}

seayon::model::model(const std::vector<int> layout, const std::vector<ActivFunc> a, int seed, std::string logfolder)
	: logfolder(logfolder.size() > 0 ? ((logfolder.back() == '\\' || logfolder.back() == '/') ? logfolder : logfolder.append("/")) : logfolder),
	xsize(layout.front()),
	ysize(layout.back())
{
	if (layout.size() != a.size() + 1)
	{
		printf("--- error: layer and activation array not matching ---\n");
		return;
	}

	if (seed < 0)
	{
		std::random_device rd;
		seed = rd();

		printf("Generating with Seed: %i\n", seed);
	}

	this->seed = seed;
	std::mt19937 gen(seed);

	layers.reserve(layout.size());

	layers.emplace_back(0, layout[0], ActivFunc::LINEAR, gen);
	for (int l2 = 1; l2 < layout.size(); ++l2)
	{
		const int l1 = l2 - 1;
		layers.emplace_back(layout[l1], layout[l2], a[l1], gen);
	}
}

seayon::model::model(const model_parameters& para) : model(para.layout, para.a, para.seed, para.logfolder)
{
}