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
#include <windows.h>

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

typedef float(*ActivFunc_t)(const float&);

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
	STOCHASTIC,
	MINI_BATCH,
	ADAM
};

template <int INPUTS, int OUTPUTS>
struct trainingdata
{
	struct sample
	{
		float inputs[INPUTS]{};
		float outputs[OUTPUTS]{};
	};

private:
	sample* samples = nullptr;
	size_t sampleCount = 0;
	bool manageMemory = true;

public:
	trainingdata()
	{
	}

	trainingdata(size_t reserved)
	{
		reserve(reserved);
	}

	trainingdata(std::initializer_list<sample> list)
	{
		reserve(list.size());

		int i = 0;
		for (auto elem = list.begin(); elem != list.end(); ++elem)
		{
			samples[i++] = *elem;
		}
	}

	trainingdata(sample* samples, const size_t sampleCount, const bool manageMemory)
	{
		this->samples = samples;
		this->sampleCount = sampleCount;
		this->manageMemory = manageMemory;
	}

	inline void reserve(const size_t reserved)
	{
		this->~trainingdata();

		samples = new sample[reserved];
		sampleCount = reserved;
	}

	inline size_t size() const
	{
		return sampleCount;
	}

	inline sample& operator[](const int i) const
	{
		return samples[i];
	}

	~trainingdata()
	{
		if (manageMemory)
		{
			if (samples != nullptr)
				delete[] samples;
		}
	}
};

// Open source Neural Network library in C++ with lots of easy to use features. Copyright by Dean Schneider (deanqx, Sawey)
class seayon
{
public:
	struct layer
	{
	protected:
		const bool manageMemory;
	public:
		const ActivFunc func;
		ActivFunc_t activation;
		ActivFunc_t derivative;

		const int nCount;
		const int wCount;

		/**
		 * Goes from second to first
		 * @tparam layers[l2].weights[n2 * n1Count + n1]
		 */
		float* const weights; // TODO replace with: layers[l2].weights[n1 * n2Count + n2]
		float* const neurons;
		float* const biases;

		// layer(): manageMemory(false), func(ActivFunc::SIGMOID), nCount(0), wCount(0), neurons(nullptr), biases(nullptr), weights(nullptr)
		// {
		// }

		layer(const ActivFunc func,
			ActivFunc_t activation,
			ActivFunc_t derivative,
			float* const neurons,
			float* const biases,
			float* const weights,
			const int nCount,
			const int wCount,
			const bool manageMemory):
			func(func),
			activation(activation),
			derivative(derivative),
			neurons(neurons),
			biases(biases),
			weights(weights),
			nCount(nCount),
			wCount(wCount),
			manageMemory(manageMemory)
		{
		}

		layer(const int PREVIOUS, const int NEURONS, const ActivFunc func): nCount(NEURONS), wCount(NEURONS* PREVIOUS), func(func),
			manageMemory(true), neurons(new float[nCount]()), biases(wCount > 0 ? new float[nCount]() : nullptr), weights(wCount > 0 ? new float[wCount]() : nullptr)
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

		~layer()
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
	};
protected:
	const bool manageMemory;
	const int seed;
	const bool printcost;
	const std::string logfolder;

public:
	const int layerCount;
	layer* const layers;

	seayon(layer* layers,
		const int layerCount,
		const bool printcost,
		const int seed,
		const std::string logfolder,
		const bool manageMemory):
		layers(layers),
		layerCount(layerCount),
		seed(seed),
		printcost(printcost),
		logfolder(logfolder),
		manageMemory(manageMemory)
	{
	}

	/**
	 * Creates network where every neuron is connected to each neuron in the next layer.
	 * @param layerCount Starts with the input layer (Minimum 2 layers)
	 * @param ActivFunc Activation function for all neurons.
	 */
	seayon(const std::vector<int> layout, const std::vector<ActivFunc> a, int seed = -1, const bool printcost = true, std::string logfolder = std::string())
		: manageMemory(true), seed(seed), printcost(printcost),
		logfolder(logfolder.size() > 0 ? ((logfolder.back() == '\\' || logfolder.back() == '/') ? logfolder : logfolder.append("/")) : logfolder),
		layerCount(layout.size()), layers((layer*)malloc(layerCount * sizeof(layer)))
	{
		if (seed < 0)
			srand((unsigned int)time(NULL));
		else
			srand(seed);

		new (&layers[0]) layer(0, layout[0], a[0]);
		for (int l2 = 1; l2 < layerCount; ++l2)
		{
			const int l1 = l2 - 1;

			new (&layers[l2]) layer(layout[l1], layout[l2], a[l2]);
		}
	}

	~seayon()
	{
		if (manageMemory)
		{
			for (int i = 0; i < layerCount; ++i)
				layers[i].~layer();

			free(layers);
		}
	}

	// Saves network to a .nn file
	size_t save(std::ofstream& file)
	{
		std::vector<char> buffer;
		size_t buffersize = save(buffer);

		file.write(buffer.data(), buffersize);
		if (file.fail())
			buffersize = 0;

		file.flush();

		return buffersize;
	}
	size_t save(std::vector<char>& buffer)
	{
		size_t buffersize = 0;
		std::vector<size_t> nSize(layerCount);
		std::vector<size_t> wSize(layerCount);
		for (int i = 1; i < layerCount; ++i)
		{
			nSize[i] = sizeof(float) * layers[i].nCount;
			wSize[i] = sizeof(float) * layers[i].wCount;
			buffersize += nSize[i] + wSize[i];
		}
		buffer.resize(buffersize);

		char* pointer = buffer.data();
		for (int i = 1; i < layerCount; ++i)
		{
			memcpy(pointer, layers[i].weights, wSize[i]);
			pointer += wSize[i];
			memcpy(pointer, layers[i].biases, nSize[i]);
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

			std::vector<char> buffer(N);
			file.read(buffer.data(), N);
			load(buffer.data());

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
		std::vector<size_t> nSize(layerCount);
		std::vector<size_t> wSize(layerCount);
		for (int i = 1; i < layerCount; ++i)
		{
			nSize[i] = sizeof(float) * layers[i].nCount;
			wSize[i] = sizeof(float) * layers[i].wCount;
		}

		char* pointer = buffer;
		for (int i = 1; i < layerCount; ++i)
		{
			memcpy(layers[i].weights, pointer, wSize[i]);
			pointer += wSize[i];
			memcpy(layers[i].biases, pointer, nSize[i]);
			pointer += nSize[i];
		}
	}
	inline void copy(seayon& to) const
	{
		for (int l = 1; l < layerCount; ++l)
		{
			memcpy(to.layers[l].biases, layers[l].biases, layers[l].nCount * sizeof(float));
			memcpy(to.layers[l].weights, layers[l].weights, layers[l].wCount * sizeof(float));
		}
	}
	/**
	 * Combines two networks with the average values.
	 * @param with List of networks
	 * @param count How many networks
	 */
	void combine(seayon* with, int count)
	{
		for (int l2 = 1; l2 < layerCount; ++l2)
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
	bool equals(seayon& second)
	{
		bool equal = true;

		for (int i = 1; equal && i < layerCount; ++i)
		{
			for (int w = 0; equal && w < layers[i].wCount; ++w)
				equal = (layers[i].weights[w] == second.layers[i].weights[w]);

			for (int n = 0; equal && n < layers[i].nCount; ++n)
				equal = (layers[i].biases[n] == second.layers[i].biases[n]);
		}

		return equal;
	}

	// Calculates network outputs
	template <int INPUTS, int OUTPUTS>
	inline float* pulse(const typename trainingdata<INPUTS, OUTPUTS>::sample& sample)
	{
		for (int n = 0; n < INPUTS; ++n)
			layers[0].neurons[n] = sample.inputs[n];

		for (int l2 = 1; l2 < layerCount; ++l2)
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

		return layers[layerCount - 1].neurons;
	}

	// Prints all values. pulse() should be called before
	void print()
	{
		HANDLE cmd = GetStdHandle(STD_OUTPUT_HANDLE);

		int normalColor;

		for (int l1 = 0; l1 < layerCount; ++l1)
		{
			const int l2 = l1 + 1;

			if (l1 == 0)
			{
				normalColor = 7;
				SetConsoleTextAttribute(cmd, 7);

				printf("\n  Input Layer:\n");
			}
			else if (l1 == layerCount - 1)
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

			size_t largest = std::max_element(layers[l1].neurons, layers[l1].neurons + layers[l1].nCount) - layers[l1].neurons;
			for (int n1 = 0; n1 < layers[l1].nCount; ++n1)
			{
				printf("\t\tNeuron[%02i]   ", n1);
				if (l1 == layerCount - 1)
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

				if (l2 < layerCount)
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
	template <int INPUTS, int OUTPUTS>
	float print(trainingdata<INPUTS, OUTPUTS>& data, int sample)
	{
		pulse<INPUTS, OUTPUTS>(data[sample]);
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

		int l = layerCount - 1;

		SetConsoleTextAttribute(cmd, 11);
		printf("  Output Layer:\n");

		size_t largest = std::max_element(layers[l].neurons, layers[l].neurons + layers[l].nCount) - layers[l].neurons;
		for (int n = 0; n < layers[l].nCount; ++n)
		{
			printf("\t\tNeuron[%02i]   ", n);
			if (l == layerCount - 1)
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
	template <int INPUTS, int OUTPUTS>
	float printo(const trainingdata<INPUTS, OUTPUTS>& data, const int sample)
	{
		pulse<INPUTS, OUTPUTS>(data[sample]);
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
	template <int INPUTS, int OUTPUTS>
	float cost(const typename trainingdata<INPUTS, OUTPUTS>::sample& sample)
	{
		pulse<INPUTS, OUTPUTS>(sample);

		const int LASTL = layerCount - 1;

		float c = 0;
		for (int n = 0; n < OUTPUTS; ++n)
		{
			const float x = layers[LASTL].neurons[n] - sample.outputs[n];
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
	template <int INPUTS, int OUTPUTS>
	float cost(const trainingdata<INPUTS, OUTPUTS>& data)
	{
		if (!check(data))
		{
			printf("\tCurrupt training data!\n");
			return .0f;
		}

		float c = 0;
		for (int i = 0; i < data.size(); ++i)
		{
			c += cost<INPUTS, OUTPUTS>(data[i]);
		}

		return c / (float)data.size();
	}
	/**
	 * Describes how often the network would choose the right neurons to fire (neuron value >= 0.5).
	 * @param inputs Set values of the input layer
	 * @param outputs This are the optimal outputs
	 * @return 1.0 = 100%; 0.5f = 50%; 0.0 = 0%
	 */
	template <int INPUTS, int OUTPUTS>
	float accruacy(const trainingdata<INPUTS, OUTPUTS>& data)
	{
		const int LASTL = layerCount - 1;

		if (!check(data))
		{
			printf("\tCurrupt training data!\n");
			return .0f;
		}

		float a = 0;
		for (int i = 0; i < data.size(); ++i)
		{
			pulse<INPUTS, OUTPUTS>(data[i]);

			if (std::max_element(layers[LASTL].neurons, layers[LASTL].neurons + layers[LASTL].nCount) - layers[LASTL].neurons == std::max_element(data[i].outputs, data[i].outputs + OUTPUTS) - data[i].outputs)
			{
				++a;
			}
		}

		return a / (float)data.size();
	}

	/**
	 * Backpropagation is an efficient training algorithm which works for many cases.
	 * @param runCount How many iterations. Tip: Start small with like 5 - 20
	 * @param print Writes detailed information about the progress to the console.
	 * @param learningRate Large numbers will take less runs but can "over shoot" the right value.
	 * @param momentum Large numbers will take less runs but can "over shoot" the right value.
	 */
	template <int INPUTS, int OUTPUTS, int T_INPUTS, int T_OUTPUTS>
	void fit(int max_iterations, const trainingdata<INPUTS, OUTPUTS>& traindata, const trainingdata<T_INPUTS, T_OUTPUTS>& testdata,
		Optimizer optimizer = Optimizer::STOCHASTIC, float learningRate = 0.03f, float momentum = 0.1f, int total_threads = 32)
	{
		if (!check(traindata) || !check(testdata))
		{
			printf("\tCurrupt training data!\n");
			return;
		}

		if (optimizer == Optimizer::STOCHASTIC)
		{
			stochastic(max_iterations, learningRate, momentum, traindata, testdata);
		}
		else
		{
			if (learningRate <= 0.0f)
				learningRate = 0.0001f;

			if (momentum <= 0.0f)
				momentum = 0.0001f;

			const int batch_size = traindata.size() / total_threads;
			const int unused = traindata.size() - batch_size * total_threads;

			printf("Mini batches launching with:\n");
			printf("%i batches | ", total_threads);
			printf("%i samples per batch | ", batch_size);
			printf("%i/%i unused samples\n", unused, traindata.size());

			if (total_threads > traindata.size() / batch_size || total_threads < 0)
				total_threads = 1;

			if (optimizer == Optimizer::MINI_BATCH)
			{
				mini_batch(max_iterations, learningRate, momentum, batch_size, total_threads, traindata, testdata);
			}
			else if (optimizer == Optimizer::ADAM)
			{
			}
		}
	}

protected:
	// Returns false if training data is currupt (quickcheck)
	template <int INPUTS, int OUTPUTS>
	inline bool check(const trainingdata<INPUTS, OUTPUTS>& data) const
	{
		return layers[0].nCount == INPUTS && layers[layerCount - 1].nCount == OUTPUTS;
	}

	template <int INPUTS, int OUTPUTS>
	class fitlog
	{
		seayon& parent;
		const int sampleCount;
		const trainingdata<INPUTS, OUTPUTS>& testdata;
		const int max_iterations;
		const bool printcost;

		std::unique_ptr<std::ofstream> file{};
		size_t lastLogLenght = 0;
		int lastLogAt = 0;
		std::chrono::high_resolution_clock::time_point overall;
		std::chrono::high_resolution_clock::time_point sampleTimeLast;
		std::chrono::high_resolution_clock::time_point last;

		inline void resolveTime(long long seconds, int* resolved) const
		{
			resolved[0] = (int)(seconds / 3600LL);
			seconds -= 3600LL * (long long)resolved[0];
			resolved[1] = (int)(seconds / 60LL);
			seconds -= 60LL * (long long)resolved[1];
			resolved[2] = (int)(seconds);
		}

	public:
		void log(int run)
		{
			auto now = std::chrono::high_resolution_clock::now();
			std::chrono::microseconds sampleTime = std::chrono::duration_cast<std::chrono::microseconds>(now - sampleTimeLast);
			if (sampleTime.count() > 1000000LL || run == max_iterations)
			{
				sampleTimeLast = now;
				std::chrono::milliseconds elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last);
				std::chrono::seconds runtime = std::chrono::duration_cast<std::chrono::seconds>(now - overall);

				float c = -1.0f;
				if (printcost)
					c = parent.cost(testdata);

				float progress = (float)run * 100.0f / (float)max_iterations;

				int samplesPerSecond = 0;
				if (run > 0)
				{
					if (run > lastLogAt)
						sampleTime /= run - lastLogAt;
					if (sampleTime.count() < 1)
						samplesPerSecond = -1;
					else
						samplesPerSecond = (int)((int64_t)sampleCount * 1000LL / sampleTime.count());
				}

				int runtimeResolved[3];
				resolveTime(runtime.count(), runtimeResolved);

				if (run > lastLogAt)
					elapsed /= run - lastLogAt;
				elapsed *= max_iterations - run;

				std::chrono::seconds eta = std::chrono::duration_cast<std::chrono::seconds>(elapsed);

				int etaResolved[3];
				resolveTime(eta.count(), etaResolved);

				std::stringstream message;
				message << std::setw(4) << "Progress: " << std::fixed << std::setprecision(1) << progress << "% " << std::setw(9)
					<< samplesPerSecond << "k Samples/s " << std::setw(13)
					<< "Runtime: " << runtimeResolved[0] << "h " << runtimeResolved[1] << "m " << runtimeResolved[2] << "s " << std::setw(9)
					<< "ETA: " << etaResolved[0] << "h " << etaResolved[1] << "m " << etaResolved[2] << "s" << std::setw(9);

				if (c > -1.0f)
					message << "Cost: " << std::fixed << std::setprecision(4) << c;

				std::cout << std::string(lastLogLenght, '\b') << message.str();
				lastLogLenght = message.str().length();

				if (file.get() != nullptr)
					*file << progress << ',' << samplesPerSecond << ',' << runtime.count() << ',' << eta.count() << ',' << c << '\n';

				lastLogAt = run;
				last = std::chrono::high_resolution_clock::now();
			}
		}

		fitlog(seayon& parent,
			const int& sampleCount,
			const trainingdata<INPUTS, OUTPUTS>& testdata,
			const int& max_iterations,
			const bool& printcost,
			const std::string& logfolder):
			parent(parent),
			sampleCount(sampleCount),
			testdata(testdata),
			max_iterations(max_iterations),
			printcost(printcost)
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

				file.reset(new std::ofstream(path));
				*file << "Progress,SamplesPer(seconds),Runtime(seconds),ETA(seconds),Cost" << std::endl;
			}

			printf("\n");
			overall = std::chrono::high_resolution_clock::now();
			last = overall;
			sampleTimeLast = overall;
			log(0);
		}
	};

private:
	template <int INPUTS, int OUTPUTS>
	struct backprop_matrix
	{
		struct batch
		{
			struct layer
			{
				std::vector<float> deltas;
				std::vector<float> last_gb;
				std::vector<float> last_gw;
				std::vector<float> bias_gradients;
				std::vector<float> weight_gradients;

				const int nCount;
				const int wCount;

				layer(const int& nCount, const int& wCount)
					: nCount(nCount), wCount(wCount)
				{
					deltas.resize(nCount);
					last_gb.resize(nCount);
					last_gw.resize(wCount);
					bias_gradients.resize(nCount);
					weight_gradients.resize(wCount);
				}
			};

			std::unique_ptr<seayon> net;
			std::vector<layer> layers;

			const float sample_share;

			batch(const seayon& main, const float& sample_share)
				: sample_share(sample_share)
			{
				std::vector<int> layout(main.layerCount);
				std::vector<ActivFunc> a(main.layerCount);
				for (int l = 0; l < main.layerCount; ++l)
				{
					layout[l] = main.layers[l].nCount;
					a[l] = main.layers[l].func;
				}

				net.reset(new seayon(layout, a, main.seed, main.printcost, main.logfolder));
				main.copy(*net.get());

				layers.reserve(main.layerCount);

				layers.emplace_back(0, 0);
				for (int i = 1; i < main.layerCount; ++i)
				{
					layers.emplace_back(main.layers[i].nCount, main.layers[i].wCount);
				}
			}
		};

		std::vector<const typename trainingdata<INPUTS, OUTPUTS>::sample*> sample_pointers;
		std::vector<batch> batches;
		const int batch_count;
		const int layerCount;

		backprop_matrix(const float& sample_share, const int& sampleCount, const int& batch_count, const seayon& main, const trainingdata<INPUTS, OUTPUTS>& traindata)
			: batch_count(batch_count), layerCount(main.layerCount)
		{
			sample_pointers.resize(sampleCount);
			batches.reserve(batch_count);

			const typename trainingdata<INPUTS, OUTPUTS>::sample* sample = &traindata[0];
			for (int i = 0; i < sampleCount; ++i)
			{
				sample_pointers[i] = sample + i;
			}

			for (int i = 0; i < batch_count; ++i)
			{
				batches.emplace_back(main, sample_share);
			}
		}

		inline void apply(seayon& main)
		{
			// const float batch_share = 1.0f / batch_count;

			for (int b = 0; b < batch_count; ++b)
			{
				for (int l = 1; l < main.layerCount; ++l)
				{
					for (int n = 0; n < main.layers[l].nCount; ++n)
					{
						main.layers[l].biases[n] += batches[b].layers[l].bias_gradients[n];
					}

					for (int w = 0; w < main.layers[l].wCount; ++w)
					{
						main.layers[l].weights[w] += batches[b].layers[l].weight_gradients[w];
					}

					memset(batches[b].layers[l].bias_gradients.data(), 0, main.layers[l].nCount * sizeof(float));
					memset(batches[b].layers[l].weight_gradients.data(), 0, main.layers[l].wCount * sizeof(float));
				}
			}

			for (int b = 0; b < batch_count; ++b)
			{
				main.copy(*(batches[b].net.get()));
			}
		}

		void shuffle()
		{
			std::random_device rm_seed;
			std::shuffle(sample_pointers.begin(), sample_pointers.end(), std::mt19937(rm_seed()));
		}
	};

	template <int INPUTS, int OUTPUTS>
	static inline void backprop(const float& n, const float& m,
		typename backprop_matrix<INPUTS, OUTPUTS>::batch& thread, const typename trainingdata<INPUTS, OUTPUTS>::sample& sample)
	{
		seayon& net = *thread.net.get();
		const int LASTL = net.layerCount - 1;

		net.pulse<INPUTS, OUTPUTS>(sample);

		{
			const int& ncount = net.layers[LASTL].nCount;
			const auto& deri = net.layers[LASTL].derivative;

			for (int n2 = 0; n2 < ncount; ++n2)
			{
				thread.layers[LASTL].deltas[n2] = deri(net.layers[LASTL].neurons[n2]) * 2.0f * (net.layers[LASTL].neurons[n2] - sample.outputs[n2]);
			}
		}

		for (int l2 = LASTL; l2 >= 2; --l2)
		{
			const int l1 = l2 - 1;
			const int& n1count = net.layers[l1].nCount;
			const int& n2count = net.layers[l2].nCount;
			const auto& deri = net.layers[l1].derivative;

			for (int n1 = 0; n1 < n1count; ++n1)
			{
				float error = 0;
				for (int n2 = 0; n2 < n2count; ++n2)
					error += thread.layers[l2].deltas[n2] * net.layers[l2].weights[n2 * n1count + n1];

				thread.layers[l1].deltas[n1] = deri(net.layers[l1].neurons[n1]) * error;
			}
		}

		for (int l2 = LASTL; l2 >= 1; --l2)
		{
			const int l1 = l2 - 1;
			const int& n1count = net.layers[l1].nCount;
			const int& n2count = net.layers[l2].nCount;

			for (int n2 = 0; n2 < n2count; ++n2)
			{
				const float d = -thread.layers[l2].deltas[n2];

				thread.layers[l2].bias_gradients[n2] += (n * d + m * thread.layers[l2].last_gb[n2]) * thread.sample_share;
				thread.layers[l2].last_gb[n2] = d;

				const int row = n2 * n1count;
				for (int n1 = 0; n1 < n1count; ++n1)
				{
					const int windex = row + n1;
					const float gw = d * net.layers[l1].neurons[n1];

					thread.layers[l2].weight_gradients[windex] += (n * gw + m * thread.layers[l2].last_gw[windex]) * thread.sample_share;
					thread.layers[l2].last_gw[windex] = gw;
				}
			}
		}

		/*{
			auto& tlay = thread.layers[LASTL];
			const auto& lay = net.layers[LASTL];
			const auto& deri = lay.derivative;

			const int& ncount = lay.nCount;

			for (int n2 = 0; n2 < ncount; ++n2)
			{
				const float& neu = lay.neurons[n2];
				tlay.deltas[n2] = deri(neu) * 2.0f * (neu - sample.outputs[n2]);
			}
		}

		for (int l2 = LASTL; l2 >= 2; --l2)
		{
			const int l1 = l2 - 1;
			const auto& tlay = thread.layers[l2];
			const auto& lay1 = net.layers[l1];
			const auto& lay2 = net.layers[l2];
			const auto& deri = lay1.derivative;

			const int& n1count = lay1.nCount;
			const int& n2count = net.layers[l2].nCount;

			for (int n1 = 0; n1 < n1count; ++n1)
			{
				float error = 0;
				for (int n2 = 0; n2 < n2count; ++n2)
					error += tlay.deltas[n2] * lay2.weights[n2 * n1count + n1];

				thread.layers[l1].deltas[n1] = deri(lay1.neurons[n1]) * error;
			}
		}

		for (int l2 = LASTL; l2 >= 1; --l2)
		{
			const int l1 = l2 - 1;
			auto& tlay = thread.layers[l2];
			const auto& lay = net.layers[l1];

			const int& n1count = lay.nCount;
			const int& n2count = net.layers[l2].nCount;

			for (int n2 = 0; n2 < n2count; ++n2)
			{
				auto& last_gb = tlay.last_gb[n2];
				const float d = -tlay.deltas[n2];

				tlay.bias_gradients[n2] += (n * d + m * last_gb) * thread.sample_share;
				last_gb = d;

				const int row = n2 * n1count;
				for (int n1 = 0; n1 < n1count; ++n1)
				{
					const int windex = row + n1;
					auto& last_gw = tlay.last_gw[windex];
					const float gw = d * lay.neurons[n1];

					tlay.weight_gradients[windex] += (n * gw + m * last_gw) * thread.sample_share;
					last_gw = gw;
				}
			}
		}*/
	}

	template <int INPUTS, int OUTPUTS, int T_INPUTS, int T_OUTPUTS>
	void stochastic(const int& max_iterations, const float& n, const float& m, const trainingdata<INPUTS, OUTPUTS>& traindata, const trainingdata<T_INPUTS, T_OUTPUTS>& testdata)
	{
		std::vector<std::vector<float>> bias_gradients(layerCount);
		std::vector<std::vector<float>> weight_gradients(layerCount);

		for (int l = 1; l < layerCount; ++l)
		{
			bias_gradients[l].resize(layers[l].nCount);
			weight_gradients[l].resize(layers[l].wCount);
		}

		backprop_matrix<INPUTS, OUTPUTS> matrix(1.0f, traindata.size(), 1, *this, traindata);

		fitlog<T_INPUTS, T_OUTPUTS> logger(*this, traindata.size(), testdata, max_iterations, printcost, logfolder);

		for (int run = 1; run <= max_iterations; ++run)
		{
			for (int i = 0; i < traindata.size(); ++i)
			{
				backprop<INPUTS, OUTPUTS>(n, m, matrix.batches[0], traindata[i]);
				matrix.apply(*this);
			}

			logger.log(run);
		}

		printf("\n\n");
	}

	template <int INPUTS, int OUTPUTS, int T_INPUTS, int T_OUTPUTS>
	void mini_batch(const int& max_iterations, const float& n, const float& m, const int& batch_size, const int& total_threads, const trainingdata<INPUTS, OUTPUTS>& traindata, const trainingdata<T_INPUTS, T_OUTPUTS>& testdata)
	{
		const int sampleCount = batch_size * total_threads;

		std::vector<std::thread> threads(total_threads);

		backprop_matrix<INPUTS, OUTPUTS> matrix(1.0f / (float)sampleCount, sampleCount, total_threads, *this, traindata);

		fitlog<T_INPUTS, T_OUTPUTS> logger(*this, traindata.size(), testdata, max_iterations, printcost, logfolder);

		for (int run = 1; run <= max_iterations; ++run)
		{
			for (int b = 0; b < total_threads; ++b)
			{
				threads[b] = std::thread([&, b, n, m, batch_size]
					{
						const int begin = b * batch_size;
						const int end = begin + batch_size - 1;

						for (int i = begin; i <= end; ++i)
						{
							backprop<INPUTS, OUTPUTS>(n, m, matrix.batches[b], *matrix.sample_pointers[i]);
						}
					});
			}

			for (int b = 0; b < total_threads; ++b)
			{
				threads[b].join();
			}

			matrix.apply(*this);
			matrix.shuffle();

			logger.log(run);
		}

		printf("\n\n");
	}
};