#include <iostream>
#include <fstream>
#include "seayon.hpp"
#include "import.cpp"

using namespace seayon;

bool step_callback(model& main, int epoch, float l, float val_l, const dataset& traindata, const dataset& testdata)
{
	// printf("     %f%%\n", main.accruacy(testdata) * 100.0f);

	return false;
}

int main()
{
	constexpr bool load_pretrained = false;

	constexpr int epochs = 20;
	constexpr int batch_size = 16;
	constexpr bool shuffle = true;
	constexpr float steps_per_epoch = 0.9f;
	constexpr float learning_rate = 0.001f;

	std::vector<float> dropouts { };
	std::vector<int> layout = { 784, 16, 16, 10 };
	std::vector<ActivFunc> funcs = { ActivFunc::SIGMOID, ActivFunc::SIGMOID, ActivFunc::SIGMOID };
	model m(layout, funcs, 2, "../../../../examples/digit_recognition/logs");

	dataset testdata(784, 10);

	if (!ImportMnist(10000, testdata, "../../../../examples/digit_recognition/res/mnist_test.csv"))
		return 1;

	const char* saved_path = "../../../../examples/digit_recognition/saved.bin";

	if (load_pretrained)
	{
		model_parameters para;
		para.load_parameters_file(saved_path);

		model imported(para);
		imported.load_file(saved_path);

		imported.printo(testdata, 0);
	}
	else
	{
		// !!! You should download the full mnist-set !!!
		// 1. Download mnist_train.csv (from for example https://yann.lecun.com)
		// 2. Copy the header(first line) from mnist_test.csv
		// 3. Put mnist_train.csv in the "res/" folder

		const std::string traindata_path("../../../../examples/digit_recognition/res/mnist_train.csv");
		std::ifstream exists(traindata_path);
		if (exists.good() && true)
		{
			exists.close();

			dataset traindata(784, 10);
			if (!ImportMnist(60000, traindata, traindata_path))
				return 1;

			printf("\n");
			m.printo(testdata, 0);
			m.fit(traindata, testdata, epochs, batch_size, 4, shuffle, steps_per_epoch, learning_rate, dropouts, step_callback);
			m.printo(testdata, 0);
		}
		else
		{
			printf("\n");
			m.printo(testdata, 0);
			m.fit(testdata, testdata, epochs, batch_size, 1, shuffle, steps_per_epoch, learning_rate, dropouts);
			m.printo(testdata, 0);
		}

		m.save_file(saved_path);
	}

	return 0;
}