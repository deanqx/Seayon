#include "../seayon.hpp"

inline float randf(float min, float max)
{
	return min + (float)rand() / (float)(RAND_MAX / (max - min));
}

seayon::model::layer::layer(const int PREVIOUS, const int NEURONS, const ActivFunc func)
	: nCount(NEURONS), wCount(NEURONS* PREVIOUS), func(func), manageMemory(true),
	neurons(new float[nCount]()), biases(wCount > 0 ? new float[nCount]() : nullptr), weights(wCount > 0 ? new float[wCount]() : nullptr)
{
	if (wCount > 0)
	{
		for (int i = 0; i < wCount; ++i)
		{
			weights[i] = randf(-1.0f, 1.0f);
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

seayon::model::layer::~layer()
{
	if (manageMemory)
	{
		delete[] neurons;
		if (wCount > 0)
		{
			delete[] biases;
			delete[] weights;
		}
	}
}

seayon::model::model(const std::vector<int> layout, const std::vector<ActivFunc> a, int seed, bool printloss, std::string logfolder)
	: manageMemory(true),
	seed(seed < 0 ? (unsigned int)time(NULL) : seed),
	printloss(printloss),
	logfolder(logfolder.size() > 0 ? ((logfolder.back() == '\\' || logfolder.back() == '/') ? logfolder : logfolder.append("/")) : logfolder),
	xsize(layout.front()),
	ysize(layout.back()),
	layerCount((int)layout.size()),
	layers((layer*)malloc(layerCount * sizeof(layer)))
{
	if (layout.size() != a.size() + 1)
	{
		printf("--- error: layer and activation array not matching ---\n");
		return;
	}

	if (seed < 0)
		printf("Generating with Seed: %i\n", this->seed);

	srand(this->seed);

	new (&layers[0]) layer(0, layout[0], ActivFunc::LINEAR);
	for (int l2 = 1; l2 < layerCount; ++l2)
	{
		const int l1 = l2 - 1;

		new (&layers[l2]) layer(layout[l1], layout[l2], a[l1]);
	}
}

seayon::model::~model()
{
	if (manageMemory)
	{
		for (int i = 0; i < layerCount; ++i)
		{
			layers[i].~layer();
		}
		printf("\n");

		free(layers);
	}
}