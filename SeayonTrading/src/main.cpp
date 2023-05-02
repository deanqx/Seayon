#include "seayon.hpp"
#include "import.cpp"

int main()
{
	constexpr bool load = false;
	constexpr bool big = true;
	constexpr bool reload = true;

	constexpr int runCount = 100000;
	constexpr float learningRate = 0.0001f;
	constexpr float momentum = 0.1f;
	constexpr int thread_count = 19;
	constexpr int in = 167; // last 24 hours highest - current

	// All values are oriented by the current market position
	std::vector<int> layout =
	{
		in,
		24,
		24,
		5		// The next 5 minutes highest
	};
	std::vector<ActivFunc> funcs = { ActivFunc::RELU, ActivFunc::RELU, ActivFunc::RELU };
	seayon nn(layout, funcs, -1, true);

	trainingdata<in, 5> traindata;
	trainingdata<in, 5> testdata;

	if (big)
	{
		if (reload)
		{
			if (!reimport(-1, traindata, "../../../../SeayonTrading/res/EURUSD", 3))
				return 1;
		}
		else
		{
			if (!import(33048, traindata, "../../../../SeayonTrading/res/EURUSD", 3))
				return 1;
		}
	}
	else
	{
		if (!reimport(1000, traindata, "../../../../SeayonTrading/res/EURUSD", 2))
			return 1;
	}

	if (!reimport(-1, testdata, "../../../../SeayonTrading/res/EURUSD_test", 2))
		return 1;

	float max = traindata.max();
	float min = traindata.min();
	// traindata.normalize(max, min);
	// testdata.normalize(max, min);
	// traindata.shuffle();
	// testdata.shuffle();

	nn.printo(testdata, 0);

	if (load)
	{
		std::ifstream file("../../../../SeayonTrading/res/bot.bin", std::ios::binary);
		nn.load(file);
	}
	else
	{
		nn.fit(runCount, traindata, testdata, Optimizer::STOCHASTIC, learningRate, momentum, thread_count);

		std::ofstream file("../../../../SeayonTrading/res/bot.bin", std::ios::binary);
		nn.save(file);
	}

	nn.printo(testdata, 0);

	printf("\n");

	// c = (out - opt)^2
	// c = (1.10206 - 1.10206)^2 = 0
	// c = (1.10200 - 1.10206)^2 = 0.0000000036
	// nc = (1.10200 - 1.10206)^2 = 0.0000000036
	// #c = out - opt
	// out = #c + opt
	// out = #0.0000000036 + 1.10206 = 1.10212
	// _out = out / origin
	// _out = #c + opt / 1.10206
	// _out = 1.10212 / 1.10206 = 1.0000544
	nn.pulse<in, 5>(testdata[0]);
	const std::vector<float> denorm = nn.denormalized(max, min);
	const float* out = nn.layers[nn.layerCount - 1].neurons;
	const float range = max - min;
	const float origin = 1.10206f;
	printf("expected:        { %f, %f, %f, %f, %f }\n", testdata[0].outputs[0], testdata[0].outputs[1], testdata[0].outputs[2], testdata[0].outputs[3], testdata[0].outputs[4]);
	printf("denormalized:    { %f, %f, %f, %f, %f } range: %f\n", testdata[0].outputs[0] * range + min, testdata[0].outputs[1] * range + min, testdata[0].outputs[2] * range + min, testdata[0].outputs[3] * range + min, testdata[0].outputs[4] * range + min, range);
	printf("detransformed:   { %f, %f, %f, %f, %f } ==\n", (testdata[0].outputs[0] * range + min) * origin, (testdata[0].outputs[1] * range + min) * origin, (testdata[0].outputs[2] * range + min) * origin, (testdata[0].outputs[3] * range + min) * origin, (testdata[0].outputs[4] * range + min) * origin);
	printf("original:        { %f, %f, %f, %f, %f } origin: %f\n", 1.10206f, 1.10206f, 1.10206f, 1.10206f, 1.10206f, origin);
	printf("-->\n");
	printf("predicted:       { %f, %f, %f, %f, %f }\n", out[0] * origin, out[1] * origin, out[2] * origin, out[3] * origin, out[4] * origin);
	printf("predicted(norm): { %f, %f, %f, %f, %f }\n", denorm[0] * origin, denorm[1] * origin, denorm[2] * origin, denorm[3] * origin, denorm[4] * origin);
	printf("output:          { %f, %f, %f, %f, %f }\n\n", out[0], out[1], out[2], out[3], out[4]);

	return 0;
}