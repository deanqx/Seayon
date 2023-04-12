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
#include <algorithm>
#include <random>
#include <thread>
// TODO Add linux support
#include <windows.h>

#ifndef NDEBUG
#include "timez.hpp"
#endif

float Sigmoid(const float& z)
{
	return 1.0f / (1.0f + exp(-z));
}
float dSigmoid(const float& a)
{
	return a * (1.0f - a);
}
float Tanh(const float& z)
{
	float x0 = exp(z);
	float x1 = exp(-z);
	return (x0 - x1) / (x0 + x1);
}
float dTanh(const float& a)
{
	float t = Tanh(a);
	return 1 - t * t;
}
float ReLu(const float& z)
{
	return (z < 0.0f ? 0.0f : z);
}
float dReLu(const float& a)
{
	return (a < 0.0f ? 0.0f : 1.0f);
}
float LeakyReLu(const float& z)
{
	float x = 0.1f * z;
	return (x > z ? x : z);
}
float dLeakyReLu(const float& a)
{
	return (a > 0.0f ? 1.0f : 0.01f);
}

inline float randf(float min, float max)
{
	return min + (float)rand() / (float)(RAND_MAX / (max - min));
}

enum class ActivFunc
{
	SIGMOID,
	TANH,
	RELU,
	LEAKYRELU
};

enum class Optimizer
{
	AUTO,
	STOCHASTIC,
	MINI_BATCH,
	ADAM
};

struct layer
{
	int nCount;
	int wCount;

	ActivFunc func;
	float (*activation)(const float& z); // TODO inline
	float (*derivative)(const float& a); // WARN same function over multiple threads

	/**
	 * Goes from second to first
	 * @tparam layers[l2].weights[n2 * n1Count + n1]
	 */
	std::vector<float> weights;
	std::vector<float> neurons;
	std::vector<float> biases;

	void create(const int PREVIOUS, const int NEURONS, const ActivFunc func)
	{
		this->func = func;

		nCount = NEURONS;
		wCount = NEURONS * PREVIOUS;

		neurons.resize(nCount);
		biases.resize(nCount);
		weights.resize(wCount);

		for (int i = 0; i < wCount; ++i)
		{
			weights[i] = randf(-1.0f, 1.0f);
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
		float inputs[INPUTS]{};
		float outputs[OUTPUTS]{};
	};
	sample samples[SAMPLES]{};

	// Returns false if training data is currupt (quickcheck)
	bool check(layer* layers, int N) const
	{
		return layers[0].nCount == INPUTS && layers[N - 1].nCount == OUTPUTS;
	}
};

// TODO Rewrite Discriptions
// Open source Neural Network library in C++ with lots of easy to use features. Copyright by Dean Schneider (deanqx, Sawey)
template <int LAYERS>
class seayon
{
	const bool printEnabled;
	const bool printcost;
	const int seed;
	const std::string logfolder;

public:
	layer layers[LAYERS];

	/**
	 * Creates network where every neuron is connected to each neuron in the next layer.
	 * @param layerCount Starts with the input layer (Minimum 2 layers)
	 * @param ActivFunc Activation function for all neurons.
	 */
	seayon(const int(&layout)[LAYERS], const ActivFunc(&a)[LAYERS], const bool enablePrinting, const bool printcost, const int seed = -1, std::string logfolder = std::string())
		: printEnabled(enablePrinting), printcost(printcost), seed(seed),
		logfolder(logfolder[logfolder.back()] == '\\' || logfolder[logfolder.back()] == '/' ? logfolder : logfolder.append("/"))
	{
		if (seed < 0)
			srand(rand());
		else
			srand(seed);

		layers[0].create(0, layout[0], a[0]);
		for (int l2 = 1; l2 < LAYERS; ++l2)
		{
			const int l1 = l2 - 1;

			layers[l2].create(layout[l1], layout[l2], a[l2]);
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
	size_t save(std::ofstream& file)
	{
		char* buffer;
		size_t buffersize = save(buffer);

		file.write(buffer, buffersize);
		if (file.fail())
			buffersize = 0;

		file.flush();

		delete buffer;
		return buffersize;
	}
	// Copys network to a allocated buffer and returns size of the buffer
	size_t save(char*& buffer)
	{
		size_t buffersize = 0;
		size_t nSize[LAYERS];
		size_t wSize[LAYERS];
		for (int i = 0; i < LAYERS; ++i)
		{
			nSize[i] = sizeof(float) * layers[i].nCount;
			wSize[i] = sizeof(float) * layers[i].wCount;
			buffersize += nSize[i] + wSize[i];
		}
		buffer = (char*)malloc(buffersize);

		char* pointer = buffer;
		for (int i = 0; i < LAYERS; ++i)
		{
			memcpy(pointer, &layers[i].weights[0], wSize[i]);
			pointer += wSize[i];
			memcpy(pointer, &layers[i].biases[0], nSize[i]);
			pointer += nSize[i];
		}

		return buffersize;
	}
	/**
	 * Loads .nn file
	 * @exception Currupt .nn files will throw an error!
	 */
	bool load(std::ifstream& file)
	{
		if (file.is_open())
		{
			file.seekg(0, file.end);
			int N = (int)file.tellg();
			file.seekg(0, file.beg);

			char* buffer = new char[N];
			file.read(buffer, N);
			load(buffer);

			delete[] buffer;
			return true;
		}

		return false;
	}
	/**
	 * Loads network from a std::string
	 * @exception Currupt string will throw an error!
	 */
	void load(char* buffer)
	{
		size_t nSize[LAYERS];
		size_t wSize[LAYERS];
		for (int i = 0; i < LAYERS; ++i)
		{
			nSize[i] = sizeof(float) * layers[i].nCount;
			wSize[i] = sizeof(float) * layers[i].wCount;
		}

		char* pointer = buffer;
		for (int i = 0; i < LAYERS; ++i)
		{
			memcpy(&layers[i].weights[0], pointer, wSize[i]);
			pointer += wSize[i];
			memcpy(&layers[i].biases[0], pointer, nSize[i]);
			pointer += nSize[i];
		}
	}
	inline void copy(seayon<LAYERS>& to)
	{
		for (int l = 0; l < LAYERS; ++l)
		{
			memcpy(&to.layers[l].biases[0], &layers[l].biases[0], layers[l].nCount * sizeof(float));
			memcpy(&to.layers[l].weights[0], &layers[l].weights[0], layers[l].wCount * sizeof(float));
		}
	}
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

				layers[l2].neurons[n2] = an / (count + 1);

				for (int n1 = 0; n1 < layers[l1].nCount; ++n1)
				{
					float aw = layers[l2].weights[n2 * layers[l1].nCount + n1];
					for (int i = 0; i < count; ++i)
						aw += with[i].layers[l2].weights[n2 * layers[l1].nCount + n1];

					layers[l2].weights[n2 * layers[l1].nCount + n1] = aw / (count + 1);
				}
			}
		}
		for (int n1 = 0; n1 < layers[0].nCount; ++n1)
		{
			float an = layers[0].neurons[n1];
			for (int i = 0; i < count; ++i)
				an += with[i].layers[0].neurons[n1];

			layers[0].neurons[n1] = an / (count + 1);
		}
	}
	bool equals(seayon<LAYERS>& second)
	{
		bool equal = true;

		for (int i = 0; equal && i < LAYERS; ++i)
		{
			for (int w = 0; equal && w < layers[i].wCount; ++w)
				equal = (layers[i].weights[w] == second.layers[i].weights[w]);

			for (int n = 0; equal && n < layers[i].nCount; ++n)
				equal = (layers[i].biases[n] == second.layers[i].biases[n]);
		}

		return equal;
	}

	// Calculates network outputs
	template <int SAMPLES, int INPUTS, int OUTPUTS>
	inline void pulse(const typename trainingdata<SAMPLES, INPUTS, OUTPUTS>::sample& sample)
	{
		for (int n = 0; n < INPUTS; ++n)
			layers[0].neurons[n] = sample.inputs[n];

		for (int l2 = 1; l2 < LAYERS; ++l2)
		{
			const int l1 = l2 - 1;
			const int& n1count = layers[l1].nCount;
			const int& n2count = layers[l2].nCount;
			const auto& func = layers[l2].activation;

			for (int n2 = 0; n2 < n2count; ++n2)
			{
				float z = 0;
				for (int n1 = 0; n1 < n1count; ++n1)
					z += layers[l2].weights[n2 * n1count + n1] * layers[l1].neurons[n1];
				z += layers[l2].biases[n2];

				layers[l2].neurons[n2] = func(z);
			}
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
			else if (l1 == LAYERS - 1) // TODO Use printo instead
			{
				normalColor = 11;
				SetConsoleTextAttribute(cmd, 11);

				printf("  Output Layer:\n");
			}
			else
			{
				normalColor = 8;
				SetConsoleTextAttribute(cmd, 8);

				printf("  Hidden Layer[%i]:\n", l1 - 1);
			}

			int largest = std::max_element(&layers[l1].neurons[0], &layers[l1].neurons[0] + layers[l1].nCount) - &layers[l1].neurons[0];
			for (int n1 = 0; n1 < layers[l1].nCount; ++n1)
			{
				printf("\t\tNeuron[%02i]   ", n1);
				if (l1 == LAYERS - 1)
				{
					if (n1 == largest)
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

				if (l2 < LAYERS)
					for (int n2 = 0; n2 < layers[l2].nCount; ++n2)
					{
						printf("\t\t  Weight[%02i] ", n2);
						float& w = layers[l2].weights[n2 * layers[l1].nCount + n1];

						if (w <= 0.0f)
						{
							SetConsoleTextAttribute(cmd, 12);
							printf("%.2f\n", w);
							SetConsoleTextAttribute(cmd, normalColor);
						}
						else
							printf("%.2f\n", w);
					}
				printf("\n");
			}
		}
		SetConsoleTextAttribute(cmd, 7);
		printf("\t-----------------------------------------------\n\n");
	}
	// Prints all values with the cost() and the accruacy().
	template <int SAMPLES, int INPUTS, int OUTPUTS>
	float print(trainingdata<SAMPLES, INPUTS, OUTPUTS>& data, int sample)
	{
		pulse<SAMPLES, INPUTS, OUTPUTS>(data.samples[sample]);
		print();

		float a = accruacy(data);
		printf("\t\tCost\t\t%.3f\n", cost(data));
		printf("\t\tAccruacy\t%.1f%%\n", accruacy(data) * 100.0f);
		printf("\t-----------------------------------------------\n\n");

		return a;
	}
	// Prints out the output layer. pulse() should be called before
	void printo()
	{
		HANDLE cmd = GetStdHandle(STD_OUTPUT_HANDLE);

		int normalColor = 11;

		int l = LAYERS - 1;

		SetConsoleTextAttribute(cmd, 11);
		printf("  Output Layer:\n");

		int largest = std::max_element(&layers[l].neurons[0], &layers[l].neurons[0] + layers[l].nCount) - &layers[l].neurons[0];
		for (int n = 0; n < layers[l].nCount; ++n)
		{
			printf("\t\tNeuron[%02i]   ", n);
			if (l == LAYERS - 1)
			{
				if (n == largest)
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
	float printo(const trainingdata<SAMPLES, INPUTS, OUTPUTS>& data, const int sample)
	{
		pulse<SAMPLES, INPUTS, OUTPUTS>(data.samples[sample]);
		printo();

		float a = accruacy(data);
		printf("\t\tCost\t\t%.3f\n", cost(data));
		printf("\t\tAccruacy\t%.1f%%\n", accruacy(data) * 100.0f);
		printf("\t-----------------------------------------------\n\n");

		return a;
	}
	/**
	 * Describes how far apart the optimal outputs from the outputs are.
	 * @param outputs This are the optimal outputs
	 * @return 0.0 = Best; 1.0 or more = Bad
	 */
	template <int SAMPLES, int INPUTS, int OUTPUTS>
	float cost(const typename trainingdata<SAMPLES, INPUTS, OUTPUTS>::sample& sample)
	{
		pulse<SAMPLES, INPUTS, OUTPUTS>(sample);

		constexpr int LASTL = LAYERS - 1;

		float c = 0;
		for (int n = 0; n < OUTPUTS; ++n)
		{
			float x = layers[LASTL].neurons[n] - sample.outputs[n];
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
	float cost(const trainingdata<SAMPLES, INPUTS, OUTPUTS>& data)
	{
		if (!data.check(&layers[0], LAYERS))
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
	float accruacy(const trainingdata<SAMPLES, INPUTS, OUTPUTS>& data)
	{
		constexpr int LASTL = LAYERS - 1;

		if (!data.check(&layers[0], LAYERS))
		{
			printf("\tCurrupt training data!\n");
			return .0f;
		}

		float a = 0;
		for (int i = 0; i < SAMPLES; ++i)
		{
			pulse<SAMPLES, INPUTS, OUTPUTS>(data.samples[i]);

			if (std::max_element(&layers[LASTL].neurons[0], &layers[LASTL].neurons[0] + layers[LASTL].nCount) - &layers[LASTL].neurons[0] == std::max_element(&data.samples[i].outputs[0], &data.samples[i].outputs[0] + OUTPUTS) - &data.samples[i].outputs[0])
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
	void fit(int runCount, const trainingdata<SAMPLES, INPUTS, OUTPUTS>& traindata, const trainingdata<T_SAMPLES, T_INPUTS, T_OUTPUTS>& testdata, Optimizer optimizer = Optimizer::AUTO, float learningRate = 0.03f, float momentum = 0.1f, int batch_size = 50)
	{
		if (!traindata.check(&layers[0], LAYERS) || !testdata.check(&layers[0], LAYERS))
		{
			if (printEnabled)
				printf("\tCurrupt training data!\n");
			return;
		}

		// TODO change runCount to max_iterations
		if (optimizer == Optimizer::AUTO)
		{
			// TODO automatic optimizer
		}
		else if (optimizer == Optimizer::STOCHASTIC)
		{
			stochastic(runCount, learningRate, momentum, traindata, testdata);
		}
		else if (optimizer == Optimizer::MINI_BATCH)
		{
			mini_batch(runCount, learningRate, momentum, batch_size, traindata, testdata);
		}
		else if (optimizer == Optimizer::ADAM)
		{
			// TODO ADAM
		}
	}

private:
	void resolveTime(int seconds, int* resolved)
	{
		resolved[0] = seconds / 3600;
		seconds -= 3600 * resolved[0];
		resolved[1] = seconds / 60;
		seconds -= 60 * resolved[1];
		resolved[2] = seconds;
	}

	void log(std::ofstream* file, const int& run, const int& runCount, const int& sampleCount, const int& runtime, std::chrono::microseconds elapsed, std::chrono::microseconds sampleTime, const float& c)
	{
		float progress = (float)run * 100.0f / (float)runCount;

		int samplesPerSecond = 0;
		if (run > 0)
		{
			samplesPerSecond = (int)((int64_t)sampleCount * 1000000LL / sampleTime.count());
		}

		int runtimeResolved[3];
		resolveTime(runtime, runtimeResolved);

		elapsed *= runCount - run;
		int eta = (int)std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
		int etaResolved[3];
		resolveTime(eta, etaResolved);

		std::cout << "\r" << std::setw(4) << "Progress: " << std::fixed << std::setprecision(1) << progress << "% " << std::setw(9)
			<< samplesPerSecond << " Samples/s " << std::setw(13)
			<< "Runtime: " << runtimeResolved[0] << "h " << runtimeResolved[1] << "m " << runtimeResolved[2] << "s " << std::setw(9)
			<< "ETA: " << etaResolved[0] << "h " << etaResolved[1] << "m " << etaResolved[2] << "s" << std::setw(9);
		if (c > -1.0f)
			std::cout << "Cost: " << std::fixed << std::setprecision(4) << c;
		std::cout << "                " << std::flush;

		if (file)
			*file << progress << "," << samplesPerSecond << "," << runtime << "," << eta << "," << c << std::endl;
	}

	void init_log(const float& c, const int& runCount, const int& SAMPLES, std::ofstream*& logfile)
	{
		if (!logfolder.empty())
		{
			std::string path(logfolder + "log.csv");
			std::ifstream exists(path);
			for (int i = 1; i < 16; ++i)
			{
				if (!exists.good())
				{
					break;
				}
				exists.close();
				path = logfolder + "log(" + std::to_string(i) + ").csv";
				exists.open(path);
			}

			logfile = new std::ofstream(path);
			*logfile << "Progress,SamplesPer(seconds),Runtime(seconds),ETA(seconds),Cost" << std::endl;
		}

		printf("\n");
		log(logfile, 0, runCount, SAMPLES, 0, std::chrono::microseconds::zero(), std::chrono::microseconds::zero(), c);
	}

	template <int SAMPLES, int INPUTS, int OUTPUTS, int T_SAMPLES, int T_INPUTS, int T_OUTPUTS>
	void stochastic(const int& runCount, const float& n, const float& m, const trainingdata<SAMPLES, INPUTS, OUTPUTS>& data, const trainingdata<T_SAMPLES, T_INPUTS, T_OUTPUTS>& testdata)
	{
		constexpr int LASTL = LAYERS - 1;
		std::ofstream* logfile = nullptr;

		if (printEnabled)
		{
			float c = -1.0f;
			if (printcost)
				c = cost(testdata);

			init_log(c, runCount, SAMPLES, logfile);
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

		auto overall = std::chrono::high_resolution_clock::now();
		auto last = std::chrono::high_resolution_clock::now();
		auto sampleTimeLast = std::chrono::high_resolution_clock::now();

		// PERF Speed becomes slower overtime
		// (could be because floats get more complex; because of momentum?!)
		for (int run = 1; run <= runCount; ++run)
		{
			for (int i = 0; i < SAMPLES; ++i)
			{
				pulse<SAMPLES, INPUTS, OUTPUTS>(data.samples[i]);

				for (int n2 = 0; n2 < layers[LASTL].nCount; ++n2)
				{
					dn[LASTL][n2] = layers[LASTL].derivative(layers[LASTL].neurons[n2]) * 2.0f * (layers[LASTL].neurons[n2] - data.samples[i].outputs[n2]);
				}

				for (int l2 = LASTL; l2 >= 2; --l2)
				{
					const int l1 = l2 - 1;
					const int& n1count = layers[l1].nCount;
					const int& n2count = layers[l2].nCount;
					const auto& deri = layers[l1].derivative;

					for (int n1 = 0; n1 < n1count; ++n1)
					{
						float error = 0;
						for (int n2 = 0; n2 < n2count; ++n2)
							error += dn[l2][n2] * layers[l2].weights[n2 * n1count + n1];

						dn[l1][n1] = deri(layers[l1].neurons[n1]) * error;
					}
				}

				for (int l2 = LASTL; l2 >= 1; --l2)
				{
					const int l1 = l2 - 1;
					const int& n1count = layers[l1].nCount;
					const int& n2count = layers[l2].nCount;

					for (int n2 = 0; n2 < n2count; ++n2)
					{
						const int row = n2 * n1count;
						const float d = -dn[l2][n2];

						layers[l2].biases[n2] += n * d + m * lastdb[l2][n2];
						lastdb[l2][n2] = d;

						for (int n1 = 0; n1 < n1count; ++n1)
						{
							const int windex = row + n1;
							const float dw = layers[l1].neurons[n1] * d;
							layers[l2].weights[windex] += n * dw + m * lastdw[l2][windex];
							lastdw[l2][windex] = dw;
						}
					}
				}
			}
			if (printEnabled)
			{
				auto now = std::chrono::high_resolution_clock::now();
				std::chrono::microseconds sampleTime = std::chrono::duration_cast<std::chrono::microseconds>(now - sampleTimeLast);
				std::chrono::seconds runtime = std::chrono::duration_cast<std::chrono::seconds>(now - overall);
				std::chrono::microseconds elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - last);
				last = now;

				float c = -1.0f;
				if (printcost)
					c = cost(testdata);

				log(logfile, run, runCount, SAMPLES, runtime.count(), elapsed, sampleTime, c);

				sampleTimeLast = std::chrono::high_resolution_clock::now();
			}
		}

		if (printEnabled)
		{
			std::cout << std::endl
				<< std::endl;
		}

		if (logfile != nullptr)
		{
			logfile->close();
			delete logfile;
		}

		for (int l = 0; l < LAYERS; ++l)
		{
			delete[] dn[l];
			delete[] lastdb[l];
			delete[] lastdw[l];
		}
	}

	template <int SAMPLES, int INPUTS, int OUTPUTS>
	void gradient_descent(const float n, const float m,
		const typename trainingdata<SAMPLES, INPUTS, OUTPUTS>::sample** samples_begin,
		const typename trainingdata<SAMPLES, INPUTS, OUTPUTS>::sample** samples_end,
		seayon<LAYERS>& net,
		std::vector<std::vector<float>>& deltas,
		std::vector<std::vector<float>>& last_gb,
		std::vector<std::vector<float>>& last_gw,
		std::vector<std::vector<float>>& bias_gradients,
		std::vector<std::vector<float>>& weight_gradients)
	{
		constexpr int LASTL = LAYERS - 1;

		for (const typename trainingdata<SAMPLES, INPUTS, OUTPUTS>::sample** sample = samples_begin; sample <= samples_end; ++sample)
		{
			net.pulse<SAMPLES, INPUTS, OUTPUTS>(**sample);

			for (int n2 = 0; n2 < net.layers[LASTL].nCount; ++n2)
			{
				const int l1 = LASTL - 1;
				const int& n1count = net.layers[l1].nCount;
				const int row = n2 * n1count;

				const float delta = net.layers[LASTL].derivative(net.layers[LASTL].neurons[n2]) * 2.0f * (net.layers[LASTL].neurons[n2] - (*sample)->outputs[n2]);
				const float gradient = -delta * n;

				deltas[LASTL][n2] = delta;
				bias_gradients[LASTL][n2] += gradient;

				for (int n1 = 0; n1 < n1count; ++n1)
					weight_gradients[LASTL][row + n1] += gradient * net.layers[l1].neurons[n1];
			}

			for (int l2 = LASTL; l2 >= 2; --l2)
			{
				const int l1 = l2 - 1;
				const int& n1count = net.layers[l1].nCount;
				const int& n2count = net.layers[l2].nCount;
				const auto& deri = net.layers[l1].derivative;

				for (int n1 = 0; n1 < n1count; ++n1)
				{
					float error = 0; // PERF
					for (int n2 = 0; n2 < n2count; ++n2)
						error += deltas[l2][n2] * net.layers[l2].weights[n2 * n1count + n1];

					deltas[l1][n1] = deri(net.layers[l1].neurons[n1]) * error;
				}
			}

			// PERF Try to combine loops
			for (int l2 = LASTL; l2 >= 1; --l2)
			{
				const int l1 = l2 - 1;
				const int& n1count = net.layers[l1].nCount;
				const int& n2count = net.layers[l2].nCount;

				for (int n2 = 0; n2 < n2count; ++n2)
				{
					const float d = -deltas[l2][n2];
					const float gradient = d * n;

					bias_gradients[l2][n2] += gradient + m * last_gb[l2][n2];
					last_gb[l2][n2] = d;

					const int row = n2 * n1count;
					for (int n1 = 0; n1 < n1count; ++n1)
					{
						const int windex = row + n1;
						const float gw = gradient * net.layers[l1].neurons[n1];

						weight_gradients[l2][windex] += gw + m * last_gw[l2][windex];
						last_gw[l2][windex] = gw;
					}
				}
			}
		}
	}

	template <int SAMPLES, int INPUTS, int OUTPUTS, int T_SAMPLES, int T_INPUTS, int T_OUTPUTS>
	void mini_batch(const int& runCount, const float& n, const float& m, const int& batch_size, const trainingdata<SAMPLES, INPUTS, OUTPUTS>& traindata, const trainingdata<T_SAMPLES, T_INPUTS, T_OUTPUTS>& testdata)
	{
		constexpr int LASTL = LAYERS - 1;
		const int batch_count = SAMPLES / batch_size;
		std::ofstream* logfile = nullptr;

		if (printEnabled)
		{
			float c = -1.0f;
			if (printcost)
				c = cost(testdata);

			init_log(c, runCount, SAMPLES, logfile);
		}

		int layout[LAYERS];
		ActivFunc a[LAYERS];
		for (int l = 0; l < LAYERS; ++l)
		{
			layout[l] = layers[l].nCount;
			a[l] = layers[l].func;
		}

		std::vector<seayon<LAYERS>> nets = std::vector<seayon<LAYERS>>(batch_count, seayon<LAYERS>(layout, a, printEnabled, printcost, seed, logfolder));
		std::vector<std::vector<std::vector<float>>> deltas(batch_count, std::vector<std::vector<float>>(LAYERS));
		std::vector<std::vector<std::vector<float>>> last_gb(batch_count, std::vector<std::vector<float>>(LAYERS));
		std::vector<std::vector<std::vector<float>>> last_gw(batch_count, std::vector<std::vector<float>>(LAYERS));

		const auto** sample_pointers = new const typename trainingdata<SAMPLES, INPUTS, OUTPUTS>::sample* [batch_count * batch_size];
		std::vector<std::vector<std::vector<float>>> bias_gradients(batch_count, std::vector<std::vector<float>>(LAYERS));
		std::vector<std::vector<std::vector<float>>> weight_gradients(batch_count, std::vector<std::vector<float>>(LAYERS));

		std::thread* myThreads = new std::thread[batch_count];

		for (int b = 0; b < batch_count; ++b)
		{
			copy(nets[b]);

			for (int i = 0; i < batch_count * batch_size; ++i)
			{
				sample_pointers[i] = &traindata.samples[i];
			}

			for (int l = 0; l < LAYERS; ++l)
			{
				deltas[b][l].resize(layers[l].nCount);
				last_gb[b][l].resize(layers[l].nCount);
				last_gw[b][l].resize(layers[l].wCount);
				bias_gradients[b][l].resize(layers[l].nCount);
				weight_gradients[b][l].resize(layers[l].wCount);
			}
		}

		auto overall = std::chrono::high_resolution_clock::now();
		auto last = std::chrono::high_resolution_clock::now();
		auto sampleTimeLast = std::chrono::high_resolution_clock::now();

		timez::init();

		for (int run = 1; run <= runCount; ++run)
		{
			// for (int b = 0; b < batch_count; ++b)
			// {
			// 	const int offset = b * batch_size;
			// 	myThreads[b] = std::thread(gradient_descent<SAMPLES, INPUTS, OUTPUTS>,
			// 		n, m,
			// 		sample_pointers + offset,
			// 		sample_pointers + offset + batch_size - 1,
			// 		nets[b], deltas[b], last_gb[b], last_gw[b], bias_gradients[b], weight_gradients[b]);
			// }

			{
				timez::perf p0("A");

				for (int b = 0; b < batch_count; ++b)
				{
					timez::perf ba("batch");

					const int offset = b * batch_size;
					gradient_descent<SAMPLES, INPUTS, OUTPUTS>(n, m,
						sample_pointers + offset,
						sample_pointers + offset + batch_size - 1,
						nets[b], deltas[b], last_gb[b], last_gw[b], bias_gradients[b], weight_gradients[b]);
				}
			}

			// for (int b = 0; b < batch_count; ++b)
			// {
			// 	myThreads[b].join();
			// }

			{
				timez::perf p1("B");

				for (int b = 0; b < batch_count; ++b)
				{
					// PERF could be paralell
					for (int l2 = LASTL; l2 >= 1; --l2)
					{
						const int l1 = l2 - 1;
						const int& n1count = layers[l1].nCount;
						const int& n2count = layers[l2].nCount;

						for (int n2 = 0; n2 < n2count; ++n2)
						{
							layers[l2].biases[n2] += bias_gradients[b][l2][n2];

							const int row = n2 * n1count;
							for (int n1 = 0; n1 < n1count; ++n1)
							{
								const int windex = row + n1;
								layers[l2].weights[windex] += weight_gradients[b][l2][windex];
							}
						}

						memset(&last_gb[b][l2][0], 0, n2count * sizeof(float));
						memset(&last_gw[b][l2][0], 0, layers[l2].wCount * sizeof(float));
						memset(&bias_gradients[b][l2][0], 0, n2count * sizeof(float));
						memset(&weight_gradients[b][l2][0], 0, layers[l2].wCount * sizeof(float));
					}
				}
			}

			{
				timez::perf p2("C");

				for (int b = 0; b < batch_count; ++b)
				{
					copy(nets[b]);
				}
			}

			{
				timez::perf p3("D");

				std::random_device rm_seed;
				std::shuffle(&sample_pointers[0], &sample_pointers[batch_count * batch_size - 1], std::mt19937(rm_seed()));
			}

			if (printEnabled)
			{
				auto now = std::chrono::high_resolution_clock::now();
				std::chrono::microseconds sampleTime = std::chrono::duration_cast<std::chrono::microseconds>(now - sampleTimeLast);
				std::chrono::seconds runtime = std::chrono::duration_cast<std::chrono::seconds>(now - overall);
				std::chrono::microseconds elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - last);
				last = now;

				float c = -1.0f;
				if (printcost)
					c = cost(testdata);

				log(logfile, run, runCount, SAMPLES, runtime.count(), elapsed, sampleTime, c);

				sampleTimeLast = std::chrono::high_resolution_clock::now();
			}
		}


		if (printEnabled)
		{
			std::cout << std::endl
				<< std::endl;
		}

		timez::print();

		if (logfile != nullptr)
		{
			logfile->close();
			delete logfile;
		}

		delete[] sample_pointers;
	}
};