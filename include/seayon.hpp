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
		float (*activation)(const float& z);
		float (*derivative)(const float& a);

		const int nCount;
		const int wCount;

		/**
		 * Goes from second to first
		 * @tparam layers[l2].weights[n2 * n1Count + n1]
		 */
		float* const weights;
		float* const neurons;
		float* const biases;

		// layer(): manageMemory(false), func(ActivFunc::SIGMOID), nCount(0), wCount(0), neurons(nullptr), biases(nullptr), weights(nullptr)
		// {
		// }

		layer(const ActivFunc func,
			float (*activation)(const float& z),
			float (*derivative)(const float& a),
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
	const bool printcost;
	const int seed;
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
		printcost(printcost),
		seed(seed),
		logfolder(logfolder),
		manageMemory(manageMemory)
	{
	}

	/**
	 * Creates network where every neuron is connected to each neuron in the next layer.
	 * @param layerCount Starts with the input layer (Minimum 2 layers)
	 * @param ActivFunc Activation function for all neurons.
	 */
	seayon(const std::vector<int> layout, const std::vector<ActivFunc> a, const bool printcost, int seed = -1, std::string logfolder = std::string())
		: manageMemory(true), printcost(printcost), seed(seed),
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
			free(layers);
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

		free(buffer);
		return buffersize;
	}
	// warning: use free()
	size_t save(char*& buffer)
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
		buffer = (char*)malloc(buffersize);

		char* pointer = buffer;
		for (int i = 1; i < layerCount; ++i)
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
			memcpy(&layers[i].weights[0], pointer, wSize[i]);
			pointer += wSize[i];
			memcpy(&layers[i].biases[0], pointer, nSize[i]);
			pointer += nSize[i];
		}
	}
	inline void copy(seayon& to)
	{
		for (int l = 1; l < layerCount; ++l)
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
	inline void pulse(const typename trainingdata<INPUTS, OUTPUTS>::sample& sample)
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

			size_t largest = std::max_element(&layers[l1].neurons[0], &layers[l1].neurons[0] + layers[l1].nCount) - &layers[l1].neurons[0];
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

		size_t largest = std::max_element(&layers[l].neurons[0], &layers[l].neurons[0] + layers[l].nCount) - &layers[l].neurons[0];
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

			if (std::max_element(&layers[LASTL].neurons[0], &layers[LASTL].neurons[0] + layers[LASTL].nCount) - &layers[LASTL].neurons[0] == std::max_element(&data[i].outputs[0], &data[i].outputs[0] + OUTPUTS) - &data[i].outputs[0])
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
	void fit(int max_iterations, const trainingdata<INPUTS, OUTPUTS>& traindata, const trainingdata<T_INPUTS, T_OUTPUTS>& testdata, Optimizer optimizer = Optimizer::STOCHASTIC, float learningRate = 0.03f, float momentum = 0.1f, int batch_size = 50, int thread_count = 32)
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
		else if (optimizer == Optimizer::MINI_BATCH)
		{
			if (batch_size > traindata.size())
				batch_size = traindata.size();

			mini_batch(max_iterations, learningRate, momentum, batch_size, thread_count, traindata, testdata);
		}
		else if (optimizer == Optimizer::ADAM)
		{
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

		std::unique_ptr<std::ofstream> file;
		size_t lastLogLenght = 0;
		int lastLogAt = 0;
		std::chrono::steady_clock::time_point overall;
		std::chrono::steady_clock::time_point sampleTimeLast;
		std::chrono::steady_clock::time_point last;

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
			if (sampleTime.count() > 1000000LL || run > max_iterations)
			{
				sampleTimeLast = now;
				std::chrono::microseconds elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - last);
				std::chrono::seconds runtime = std::chrono::duration_cast<std::chrono::seconds>(now - overall);

				if (run > max_iterations)
					run = max_iterations;

				float c = -1.0f;
				if (printcost)
					c = parent.cost(testdata);

				float progress = (float)run * 100.0f / (float)max_iterations;

				int samplesPerSecond = 0;
				if (run > 0)
				{
					sampleTime /= run - lastLogAt;
					if (sampleTime.count() < 1)
						samplesPerSecond = -1;
					else
						samplesPerSecond = (int)((int64_t)sampleCount * 1000LL / sampleTime.count());
				}

				int runtimeResolved[3];
				resolveTime(runtime.count(), runtimeResolved);

				elapsed /= run - lastLogAt;
				elapsed *= max_iterations - run;
				long long eta = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
				int etaResolved[3];
				resolveTime(eta, etaResolved);

				std::stringstream message;
				message << "\r" << std::setw(4) << "Progress: " << std::fixed << std::setprecision(1) << progress << "% " << std::setw(9)
					<< samplesPerSecond << "k Samples/s " << std::setw(13)
					<< "Runtime: " << runtimeResolved[0] << "h " << runtimeResolved[1] << "m " << runtimeResolved[2] << "s " << std::setw(9)
					<< "ETA: " << etaResolved[0] << "h " << etaResolved[1] << "m " << etaResolved[2] << "s" << std::setw(9);
				if (c > -1.0f)
					message << "Cost: " << std::fixed << std::setprecision(4) << c;

				std::cout << std::string(lastLogLenght, '\b') << message.str();
				lastLogLenght = message.str().length();

				if (file.get() == nullptr)
					*file << progress << "," << samplesPerSecond << "," << runtime.count() << "," << eta << "," << c << std::endl;

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
	template <int INPUTS, int OUTPUTS, int T_INPUTS, int T_OUTPUTS>
	void stochastic(const int& max_iterations, const float& n, const float& m, const trainingdata<INPUTS, OUTPUTS>& traindata, const trainingdata<T_INPUTS, T_OUTPUTS>& testdata)
	{
		const int LASTL = layerCount - 1;

		std::vector<std::vector<float>> dn(layerCount);
		std::vector<std::vector<float>> lastdb(layerCount);
		std::vector<std::vector<float>> lastdw(layerCount);

		for (int l = 0; l < layerCount; ++l)
		{
			dn[l].resize(layers[l].nCount);
			lastdb[l].resize(layers[l].nCount);
			lastdw[l].resize(layers[l].wCount);
		}

		fitlog<T_INPUTS, T_OUTPUTS> logger(*this, traindata.size(), testdata, max_iterations, printcost, logfolder);

		for (int run = 1; run <= max_iterations; ++run)
		{
			for (int i = 0; i < traindata.size(); ++i)
			{
				pulse<INPUTS, OUTPUTS>(traindata[i]);

				for (int n2 = 0; n2 < layers[LASTL].nCount; ++n2)
				{
					dn[LASTL][n2] = layers[LASTL].derivative(layers[LASTL].neurons[n2]) * 2.0f * (layers[LASTL].neurons[n2] - traindata[i].outputs[n2]);
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

			logger.log(run);
		}

		logger.log(max_iterations + 1);
		printf("\n\n");
	}

	template <int INPUTS, int OUTPUTS>
	static void gradient_descent(const int id,
		const typename trainingdata<INPUTS, OUTPUTS>::sample** sample_pointers, const int thread_size,
		const float n, const float m,
		seayon& net,
		std::vector<std::vector<float>>& deltas,
		std::vector<std::vector<float>>& last_gb,
		std::vector<std::vector<float>>& last_gw,
		std::vector<std::vector<float>>& bias_gradients,
		std::vector<std::vector<float>>& weight_gradients)
	{
		const int LASTL = net.layerCount - 1;
		const int offset = id * thread_size;
		const int end = offset + thread_size - 1;

		for (int i = offset; i <= end; ++i)
		{
			const typename trainingdata<INPUTS, OUTPUTS>::sample& sample = *sample_pointers[i];
			net.pulse<INPUTS, OUTPUTS>(sample);

			{
				const int l1 = LASTL - 1;
				const int& n1count = net.layers[l1].nCount;
				const int& n2count = net.layers[LASTL].nCount;

				for (int n2 = 0; n2 < n2count; ++n2)
				{
					const int row = n2 * n1count;

					const float delta = net.layers[LASTL].derivative(net.layers[LASTL].neurons[n2]) * 2.0f * (net.layers[LASTL].neurons[n2] - sample.outputs[n2]);
					const float gradient = -delta * n;

					deltas[LASTL][n2] = delta;
					bias_gradients[LASTL][n2] += gradient;

					for (int n1 = 0; n1 < n1count; ++n1)
						weight_gradients[LASTL][row + n1] += gradient * net.layers[l1].neurons[n1];
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

	template <int INPUTS, int OUTPUTS, int T_INPUTS, int T_OUTPUTS>
	void mini_batch(const int& max_iterations, const float& n, const float& m, const int& batch_size, int thread_count, const trainingdata<INPUTS, OUTPUTS>& traindata, const trainingdata<T_INPUTS, T_OUTPUTS>& testdata)
	{
		const int LASTL = layerCount - 1;
		const int batch_count = traindata.size() / batch_size;
		int per_thread = batch_count / thread_count;

		if (per_thread == 0)
		{
			per_thread = batch_count;
			thread_count = 1;
		}

		std::vector<int> layout(layerCount);
		std::vector<ActivFunc> a(layerCount);
		for (int l = 0; l < layerCount; ++l)
		{
			layout[l] = layers[l].nCount;
			a[l] = layers[l].func;
		}

		std::vector<seayon> nets;
		nets.reserve(batch_count);
		std::vector<std::vector<std::vector<float>>> deltas(batch_count, std::vector<std::vector<float>>(layerCount));
		std::vector<std::vector<std::vector<float>>> last_gb(batch_count, std::vector<std::vector<float>>(layerCount));
		std::vector<std::vector<std::vector<float>>> last_gw(batch_count, std::vector<std::vector<float>>(layerCount));

		std::vector<const typename trainingdata<INPUTS, OUTPUTS>::sample*> sample_pointers(batch_count * batch_size);
		std::vector<std::vector<std::vector<float>>> bias_gradients(batch_count, std::vector<std::vector<float>>(layerCount));
		std::vector<std::vector<std::vector<float>>> weight_gradients(batch_count, std::vector<std::vector<float>>(layerCount));

		std::vector<std::thread> threads(thread_count);

		for (int b = 0; b < batch_count; ++b)
		{
			nets.emplace_back(layout, a, printcost, seed, logfolder);
			copy(nets[b]);

			for (int i = 0; i < batch_count * batch_size; ++i)
			{
				sample_pointers[i] = &traindata[i];
			}

			for (int l = 0; l < layerCount; ++l)
			{
				deltas[b][l].resize(layers[l].nCount);
				last_gb[b][l].resize(layers[l].nCount);
				last_gw[b][l].resize(layers[l].wCount);
				bias_gradients[b][l].resize(layers[l].nCount);
				weight_gradients[b][l].resize(layers[l].wCount);
			}
		}

		fitlog<T_INPUTS, T_OUTPUTS> logger(*this, traindata.size(), testdata, max_iterations, printcost, logfolder);

		for (int run = 1; run <= max_iterations; ++run)
		{
			for (int t = 0; t < thread_count; ++t)
			{
				threads[t] = std::thread([&, t]
					{
						const int offset = t * per_thread;
						const int end = offset + per_thread - 1;

						for (int b = offset; b <= end; ++b)
						{
							gradient_descent<INPUTS, OUTPUTS>(b, sample_pointers.data(), batch_size,
								n, m, nets[b], deltas[b], last_gb[b], last_gw[b], bias_gradients[b], weight_gradients[b]);
						}
					});
			}

			for (int t = 0; t < thread_count; ++t)
			{
				threads[t].join();
			}

			for (int b = 0; b < batch_count; ++b)
			{
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

			for (int b = 0; b < batch_count; ++b)
			{
				copy(nets[b]);
			}

			std::random_device rm_seed;
			std::shuffle(&sample_pointers[0], &sample_pointers[batch_count * batch_size - 1], std::mt19937(rm_seed()));

			logger.log(run);
		}


		logger.log(max_iterations + 1);
		printf("\n\n");
	}
};