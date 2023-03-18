#pragma once

#include <math.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <array>
#include <iostream>
// TODO Add linux support
#include <windows.h>

#include "../../.modules/timez/timez.hpp"

inline float ReLu(float z)
{
	return (z < 0.0f ? 0.0f : z);
}
inline float dReLu(float a)
{
	return (a < 0.0f ? 0.0f : 1.0f);
}
inline float Sigmoid(float z)
{
	return 1.0f / (1.0f + exp(-z));
}
inline float dSigmoid(float a)
{
	return a * (1.0f - a);
}

inline float randf(float min, float max)
{
	return min + (float)rand() / (float)(RAND_MAX / (max - min));
}

struct layer
{
	int nCount;
	int wCount;

	/**
	 * Goes from second to first
	 * @tparam layers[l2].weights[n2 + n1 * n2Count]
	 */
	std::vector<float> weights;
	std::vector<float> neurons;
	std::vector<float> biases;

	void create(const int PREVIOUS, const int LAYERS)
	{
		nCount = LAYERS;
		wCount = LAYERS * PREVIOUS;
		weights.resize(wCount);
		neurons.resize(LAYERS);
		biases.resize(LAYERS);

		for (int i = 0; i < wCount; ++i)
		{
			weights[i] = randf(-2.0f, 2.0f);
		}
	}

	void clean()
	{
		neurons.clear();
		biases.clear();
		weights.clear();
	}
};

template <int SAMPLES, int INPUTS, int OUTPUTS>
struct trainingdata
{
	struct sample
	{
		std::array<float, INPUTS> inputs{}; // TOTO try std::array
		std::array<float, OUTPUTS> outputs{};
	};
	std::array<sample, SAMPLES> samples{};

	// Returns false if training data is currupt (quickcheck)
	bool check(layer* layers, int N)
	{
		return layers[0].nCount == INPUTS && layers[N - 1].nCount == OUTPUTS;
	}
};

enum class ActivFunc
{
	SIGMOID,
	RELU
};

// TODO Rewrite Discriptions
// Open source Neural Network library in C++ with lots of easy to use features. Copyright by Dean Schneider (deanqx, Sawey)
template<int LAYERS>
class seayon
{
	inline int max(float* array, int N)
	{
		int max = 0;
		for (int i = 1; i < N; ++i)
		{
			if (array[i] > array[max])
			{
				max = i;
			}
		}
		return max;
	}

	ActivFunc Activation = ActivFunc::SIGMOID;
	std::array<layer, LAYERS> layers;

	template<int SAMPLES, int INPUTS, int OUTPUTS, typename F>
	void _pulse(typename trainingdata<SAMPLES, INPUTS, OUTPUTS>::sample& sample, F func)
	{
		for (int n = 0; n < INPUTS; ++n)
			layers[0].neurons[n] = sample.inputs[n];

		const int layerCount = LAYERS - 1;
		for (int l1 = 0; l1 < layerCount; ++l1)
		{
			const int l2 = l1 + 1;

			const int& ncount = layers[l2].nCount;
			for (int n2 = 0; n2 < ncount; ++n2)
			{
				float z = 0;
				for (int n1 = 0; n1 < layers[l1].nCount; ++n1)
					z += layers[l2].weights[n2 + n1 * ncount] * layers[l1].neurons[n1];
				z += layers[l2].biases[n2];

				layers[l2].neurons[n2] = func(z);
			}
		}
	}
	void resolveTime(int seconds, int* resolved)
	{
		resolved[0] = seconds / 3600;
		seconds -= 3600 * resolved[0];
		resolved[1] = seconds / 60;
		seconds -= 60 * resolved[1];
		resolved[2] = seconds;
	}

	void log(std::ofstream* file, int run, const int runCount, const int sampleCount, int runtime, float elapsed, float c)
	{
		float progress = (float)run * 100.0f / (float)runCount;

		// TODO check
		float samplesPerSecond = (float)sampleCount / elapsed;
		if (elapsed < 0.0f)
		{
			samplesPerSecond = 0.0f;
			elapsed = 0.0f;
		}

		int runtimeResolved[3];
		resolveTime(runtime, runtimeResolved);

		int eta[3];
		resolveTime((int)(elapsed * (float)(runCount - run)), eta);

		std::ostringstream message;
		message << "\r" << std::setw(4) << "Progress: " << std::fixed << std::setprecision(1) << progress << "% " << std::setw(9)
			<< (int)samplesPerSecond << " Samples/s " << std::setw(13)
			<< "Runtime: " << runtimeResolved[0] << "h " << runtimeResolved[1] << "m " << runtimeResolved[2] << "s " << std::setw(9)
			<< "ETA: " << eta[0] << "h " << eta[1] << "m " << eta[2] << "s" << std::setw(9);

		if (c > -1.0f)
			message << "Cost: " << std::fixed << std::setprecision(4) << c;

		message << "                ";

		std::cout << message.str() << std::flush;
		if (file)
			*file << message.str() << std::endl;
	}

	template <int SAMPLES, int INPUTS, int OUTPUTS, int T_SAMPLES, int T_INPUTS, int T_OUTPUTS, typename F0, typename F1>
	void backpropagate(trainingdata<SAMPLES, INPUTS, OUTPUTS>& data, trainingdata<T_SAMPLES, T_INPUTS, T_OUTPUTS>& testdata, const int runCount, bool print, float n, float m, std::ofstream* file, const bool printcost, F0 activation, F1 derivative)
	{
		const int lastl = LAYERS - 1;

		auto overall = std::chrono::high_resolution_clock::now();
		auto last = std::chrono::high_resolution_clock::now();

		if (print)
		{
			float c = -1.0f;
			if (printcost)
				c = cost(testdata);

			printf("\n");
			log(file, 0, runCount, SAMPLES, 0, -1.0f, c);
		}

		float* dn[LAYERS];
		float* lastdb[LAYERS];
		float* lastdw[LAYERS];

		for (int l = 0; l < LAYERS; ++l)
		{
			dn[l] = new float[layers[l].nCount];
			lastdb[l] = new float[layers[l].nCount]();
			lastdw[l] = new float[layers[l].wCount]();
		}

		int l1;
		float error;
		int row;
		int windex;
		float db;
		float dw;

		// PERF Speed becomes slower overtime
		for (int run = 1; run <= runCount; ++run)
		{
			for (int i = 0; i < SAMPLES; ++i)
			{
				_pulse<SAMPLES, INPUTS, OUTPUTS>(data.samples[i], activation);

				for (int n2 = 0; n2 < layers[lastl].nCount; ++n2)
				{
					dn[lastl][n2] = derivative(layers[lastl].neurons[n2]) * 2 * (layers[lastl].neurons[n2] - data.samples[i].outputs[n2]);
				}

				for (int l2 = lastl; l2 >= 2; --l2)
				{
					l1 = l2 - 1;
					const int& ncount = layers[l2].nCount;

					for (int n1 = 0; n1 < layers[l1].nCount; ++n1)
					{
						error = 0;
						row = n1 * ncount;
						for (int n2 = 0; n2 < ncount; ++n2)
							error += dn[l2][n2] * layers[l2].weights[row + n2];

						dn[l1][n1] = derivative(layers[l1].neurons[n1]) * error;
					}
				}

				for (int l2 = lastl; l2 >= 1; --l2)
				{
					l1 = l2 - 1;

					const int& ncount = layers[l2].nCount;
					for (int n2 = 0; n2 < ncount; ++n2)
					{
						db = -dn[l2][n2];
						layers[l2].biases[n2] += n * db + m * lastdb[l2][n2];
						lastdb[l2][n2] = db;
					}

					for (int n1 = 0; n1 < layers[l1].nCount; ++n1)
					{
						row = n1 * ncount;
						for (int n2 = 0; n2 < ncount; ++n2)
						{
							windex = row + n2;
							dw = layers[l1].neurons[n1] * -dn[l2][n2];
							layers[l2].weights[windex] += n * dw + m * lastdw[l2][windex];
							lastdw[l2][windex] = dw;
						}
					}
				}
			}
			if (print) // TODO (Optional) Move printing into sample loop
			{
				std::chrono::seconds runtime = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - overall);
				std::chrono::microseconds elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - last);
				last = std::chrono::high_resolution_clock::now();

				float c = -1.0f;
				if (printcost)
					c = cost(testdata);

				log(file, run, runCount, SAMPLES, runtime.count(), (float)elapsed.count() * 1e-6f, c);
			}
		}

		if (print)
		{
			std::cout << std::endl << std::endl;
		}

		for (int l = 0; l < LAYERS; ++l)
		{
			delete[] dn[l];
			delete[] lastdb[l];
			delete[] lastdw[l];
		}
	}

public:
	/**
	 * Creates network where every neuron is connected to each neuron in the next layer.
	 * @param layerCount Starts with the input layer (Minimum 2 layers)
	 * @param ActivFunc Activation function for all neurons.
	 */
	seayon(const int* layout, const ActivFunc a = ActivFunc::SIGMOID, const int seed = -1)
	{
		Activation = a;

		if (seed < 0)
			srand(rand());
		else
			srand(seed);

		layers[0].create(0, layout[0]);
		for (int l2 = 1; l2 < LAYERS; ++l2)
		{
			const int l1 = l2 - 1;

			layers[l2].create(layout[l1], layout[l2]);
		}
	}
	void clean()
	{
		for (size_t i = 0; i < LAYERS; ++i)
		{
			layers[i].clean();
		}
	}
	// Saves network to a .nn file
	bool save(std::ofstream& file)
	{
		bool successful = false;

		char* buffer;
		size_t N = save(buffer);

		file.write(buffer, N);
		if (file)
			successful = true;

		file.flush();

		return successful;
	}
	// Copys network to a allocated buffer and returns size of the buffer
	size_t save(char*& buffer)
	{
		size_t buffersize = sizeof(layers);
		for (int i = 0; i < LAYERS; ++i)
		{
			buffersize += 2 * sizeof(float) * layers[i].nCount + sizeof(float) * layers[i].wCount;
		}
		buffer = (char*)malloc(buffersize);

		char* pointer = buffer;
		for (int i = 0; i < LAYERS; ++i)
		{
			size_t nSize = sizeof(float) * layers[i].nCount;
			size_t wSize = sizeof(float) * layers[i].wCount;

			*pointer = layers[i].nCount;
			pointer += sizeof(int);
			*pointer = layers[i].wCount;
			pointer += sizeof(int);

			memcpy(pointer, &layers[i].neurons[0], nSize);
			pointer += nSize;
			memcpy(pointer, &layers[i].biases[0], nSize);
			pointer += nSize;
			memcpy(pointer, &layers[i].weights[0], wSize); // WARN &layers[i].weights[0]
			pointer += wSize;
		}

		return buffersize;
	}
	/**
	 * Loads .nn file
	 * @exception Currupt .nn files will throw an error!
	 */
	bool load(std::ifstream& file)
	{
		bool successful = false;

		file.seekg(0, file.end);
		int N = (int)file.tellg();
		file.seekg(0, file.beg);

		char* buffer = new char[N];
		if (file.read(buffer, N))
		{
			successful = true;
		}

		load(buffer);

		return successful;
	}
	/**
	 * Loads network from a std::string
	 * @exception Currupt string will throw an error!
	 */
	void load(char* buffer)
	{
		char* pointer = buffer;
		for (int i = 0; i < LAYERS; ++i)
		{
			size_t nSize = sizeof(float) * layers[i].nCount;
			size_t wSize = sizeof(float) * layers[i].wCount;

			*pointer = layers[i].nCount;
			pointer += sizeof(int);
			*pointer = layers[i].wCount;
			pointer += sizeof(int);

			memcpy(&layers[i].neurons[0], pointer, nSize);
			pointer += nSize;
			memcpy(&layers[i].biases[0], pointer, nSize);
			pointer += nSize;
			memcpy(&layers[i].weights[0], pointer, wSize);
			pointer += wSize;
		}
	}
	// TODO void copy(seayon<LAYERS>& to);
	/**
	 * Combines two networks with the average values.
	 * @param with List of networks
	 * @param count How many networks
	 */
	void combine(seayon<LAYERS>* with, int count)
	{
		for (int l2 = 1; l2 < LAYERS; ++l2)
		{
			const int l1 = l2 - 1;

			for (int n2 = 0; n2 < layers[l2].nCount; ++n2)
			{
				float an = layers[l2].neurons[n2];
				for (int i = 0; i < count; ++i)
					an += with[i].layers[l2].neurons[n2];

				layers[l2].neurons[n2] = an / count;

				for (int n1 = 0; n1 < layers[l1].nCount; ++n1)
				{
					float aw = layers[l2].weights[n2][n1];
					for (int i = 0; i < count; ++i)
						aw += with[i].layers[l2].weights[n2][n1];

					layers[l2].weights[n2][n1] = aw / count;
				}
			}
		}
		for (int n1 = 0; n1 < layers[0].nCount; ++n1)
		{
			float an = layers[0].neurons[n1];
			for (int i = 0; i < count; ++i)
				an += with[i].layers[0].neurons[n1];

			layers[0].neurons[n1] = an / count;
		}
	}
	// TODO bool equals(seayon<LAYERS>& second);

	// Calculates network outputs
	template <int SAMPLES, int INPUTS, int OUTPUTS>
	void pulse(typename trainingdata<SAMPLES, INPUTS, OUTPUTS>::sample& sample)
	{
		if (Activation == ActivFunc::SIGMOID)
		{
			_pulse<SAMPLES, INPUTS, OUTPUTS>(sample, Sigmoid);
		}
		else if (Activation == ActivFunc::RELU)
		{
			_pulse<SAMPLES, INPUTS, OUTPUTS>(sample, ReLu);
		}
	}

	// Prints all values. pulse() should be called before
	void print()
	{
		HANDLE cmd = GetStdHandle(STD_OUTPUT_HANDLE);

		int normalColor;

		for (int l1 = 0; l1 < LAYERS; ++l1)
		{
			const int l2 = l1 + 1;

			if (l1 == 0)
			{
				normalColor = 7;
				SetConsoleTextAttribute(cmd, 7);

				printf("\n  Input Layer:\n");
			}
			else if (l1 == LAYERS - 1)
			{
				normalColor = 11;
				SetConsoleTextAttribute(cmd, 11);

				printf("  Output Layer:\n");
			}
			else
			{
				normalColor = 8;
				SetConsoleTextAttribute(cmd, 8);

				printf("  Hidden Layer[%zu]:\n", l1 - 1);
			}

			for (int n1 = 0; n1 < layers[l1].nCount; ++n1)
			{
				printf("\t\tNeuron[%02zu]   ", n1);
				if (l1 == LAYERS - 1)
				{
					if (layers[l1].neurons[n1] > 0.50f)
						SetConsoleTextAttribute(cmd, 95);
					else
						SetConsoleTextAttribute(cmd, 7);
					printf("%.2f", layers[l1].neurons[n1]);
					SetConsoleTextAttribute(cmd, normalColor);
				}
				else
					printf("%0.2f", layers[l1].neurons[n1]);

				if (l1 > 0)
				{
					if (layers[l1].biases[n1] <= 0.0f)
					{
						printf("\t\t(");
						SetConsoleTextAttribute(cmd, 12);
						printf("%0.2f", layers[l1].biases[n1]);
						SetConsoleTextAttribute(cmd, normalColor);
						printf(")\n");
					}
					else
						printf("\t\t(%0.2f)\n", layers[l1].biases[n1]);
				}
				else
					printf("\n");

				if (l1 < LAYERS - 1)
					for (int n2 = 0; n2 < layers[l2].nCount; ++n2)
					{
						printf("\t\t  Weight[%02zu] ", n2);
						if (layers[l2].weights[n2 + n1 * layers[l2].nCount] <= 0.0f)
						{
							SetConsoleTextAttribute(cmd, 12);
							printf("%.2f\n", layers[l2].weights[n2 + n1 * layers[l2].nCount]);
							SetConsoleTextAttribute(cmd, normalColor);
						}
						else
							printf("%.2f\n", layers[l2].weights[n2 + n1 * layers[l2].nCount]);
					}
				printf("\n");
			}
		}
		SetConsoleTextAttribute(cmd, 7);
		printf("\t-----------------------------------------------\n\n");
	}
	// Prints all values with the cost() and the accruacy().
	template <int SAMPLES, int INPUTS, int OUTPUTS>
	void print(trainingdata<SAMPLES, INPUTS, OUTPUTS>& data, int sample)
	{
		_pulse<SAMPLES, INPUTS, OUTPUTS>(data.samples[sample]);
		print();
		printf("\t\tCost\t\t%.3f\n", cost(data));
		printf("\t\tAccruacy\t%.1f%%\n", accruacy(data) * 100.0f);
		printf("\t-----------------------------------------------\n\n");
	}
	// Prints out the output layer. pulse() should be called before
	void printo()
	{
		HANDLE cmd = GetStdHandle(STD_OUTPUT_HANDLE);

		int normalColor = 11;

		int l = LAYERS - 1;

		SetConsoleTextAttribute(cmd, 11);
		printf("  Output Layer:\n");

		for (int n = 0; n < layers[l].nCount; ++n)
		{
			printf("\t\tNeuron[%.2zu]   ", n);
			if (l == LAYERS - 1)
			{
				if (layers[l].neurons[n] > 0.50f)
					SetConsoleTextAttribute(cmd, 95);
				else
					SetConsoleTextAttribute(cmd, 7);
				printf("%.2f", layers[l].neurons[n]);
				SetConsoleTextAttribute(cmd, normalColor);
			}
			else
				printf("%.2f", layers[l].neurons[n]);

			if (l > 0)
			{
				if (layers[l].biases[n] <= 0.0f)
				{
					printf("\t\t(");
					SetConsoleTextAttribute(cmd, 12);
					printf("%0.2f", layers[l].biases[n]);
					SetConsoleTextAttribute(cmd, normalColor);
					printf(")\n");
				}
				else
					printf("\t\t(%0.2f)\n", layers[l].biases[n]);
			}
			else
				printf("\n\n");
		}
		SetConsoleTextAttribute(cmd, 7);
		printf("\t-----------------------------------------------\n\n");
	}
	// Prints out the output layer with the cost() and the accruacy().
	template <int SAMPLES, int INPUTS, int OUTPUTS>
	void printo(trainingdata<SAMPLES, INPUTS, OUTPUTS>& data, int sample)
	{
		pulse<SAMPLES, INPUTS, OUTPUTS>(data.samples[sample]);
		printo();
		printf("\t\tCost\t\t%.3f\n", cost(data));
		printf("\t\tAccruacy\t%.1f%%\n", accruacy(data) * 100.0f);
		printf("\t-----------------------------------------------\n\n");
	}
	/**
	 * Describes how far apart the optimal outputs from the outputs are.
	 * @param outputs This are the optimal outputs
	 * @return 0.0 = Best; 1.0 or more = Bad
	 */
	template <int SAMPLES, int INPUTS, int OUTPUTS>
	float cost(typename trainingdata<SAMPLES, INPUTS, OUTPUTS>::sample& sample)
	{
		pulse<SAMPLES, INPUTS, OUTPUTS>(sample);

		const int lastl = LAYERS - 1;

		float c = 0;
		for (int n = 0; n < OUTPUTS; ++n)
		{
			float x = layers[lastl].neurons[n] - sample.outputs[n];
			c += x * x;
		}

		return c;
		// return c / (float)OUTPUTS;
	}
	/**
	 * Describes how far apart the optimal outputs from the outputs are.
	 * @param outputs This are the optimal outputs
	 * @return 0.0 = Best; 1.0 or more = Bad
	 */
	template <int SAMPLES, int INPUTS, int OUTPUTS>
	float cost(trainingdata<SAMPLES, INPUTS, OUTPUTS>& data)
	{
		if (!data.check(layers.begin(), LAYERS))
		{
			printf("\tCurrupt training data!\n");
			return .0f;
		}

		float c = 0;
		for (int i = 0; i < SAMPLES; ++i)
		{
			c += cost<SAMPLES, INPUTS, OUTPUTS>(data.samples[i]);
		}

		return c / (float)SAMPLES;
	}
	/**
	 * Describes how often the network would choose the right neurons to fire (neuron value >= 0.5).
	 * @param inputs Set values of the input layer
	 * @param outputs This are the optimal outputs
	 * @return 1.0 = 100%; 0.5f = 50%; 0.0 = 0%
	 */
	template <int SAMPLES, int INPUTS, int OUTPUTS>
	float accruacy(trainingdata<SAMPLES, INPUTS, OUTPUTS>& data)
	{
		const int lastl = LAYERS - 1;

		if (!data.check(layers.begin(), LAYERS))
		{
			printf("\tCurrupt training data!\n");
			return .0f;
		}

		float a = 0;
		for (int i = 0; i < SAMPLES; ++i)
		{
			pulse<SAMPLES, INPUTS, OUTPUTS>(data.samples[i]);

			if (max(&layers[lastl].neurons[0], layers[lastl].nCount) == max(&data.samples[i].outputs[0], OUTPUTS))
			{
				++a;
			}
		}

		return a / (float)SAMPLES;
	}

	/**
	 * Backpropagation is an efficient training algorithm which works for many cases.
	 * @param runCount How many iterations. Tip: Start small with like 5 - 20
	 * @param print Writes detailed information about the progress to the console.
	 * @param learningRate Large numbers will take less runs but can "over shoot" the right value.
	 * @param momentum Large numbers will take less runs but can "over shoot" the right value.
	 */
	template <int SAMPLES, int INPUTS, int OUTPUTS, int T_SAMPLES, int T_INPUTS, int T_OUTPUTS>
	void fit(trainingdata<SAMPLES, INPUTS, OUTPUTS>& traindata, trainingdata<T_SAMPLES, T_INPUTS, T_OUTPUTS>& testdata, int runCount, const bool print, float learningRate = 0.03f, float momentum = 0.1f, std::ofstream* logfile = nullptr, const bool printcost = false)
	{
		if (!traindata.check(layers.begin(), LAYERS) || !testdata.check(layers.begin(), LAYERS))
		{
			if (print)
				printf("\tCurrupt training data!\n");
			return;
		}

		if (Activation == ActivFunc::SIGMOID)
		{
			backpropagate(traindata, testdata, runCount, print, learningRate, momentum, logfile, printcost, Sigmoid, dSigmoid);
		}
		else if (Activation == ActivFunc::RELU)
		{
			backpropagate(traindata, testdata, runCount, print, learningRate, momentum, logfile, printcost, ReLu, dReLu);
		}
	}
};