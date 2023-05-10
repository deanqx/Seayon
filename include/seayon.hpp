#ifndef __seayon__
#define __seayon__

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

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <conio.h>

namespace seayon
{
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
		LINEAR,
		SIGMOID,
		TANH,
		RELU,
		LEAKYRELU
	};

	enum class Optimizer
	{
		// Stochastic Gradient Descent
		STOCHASTIC,
		// ADAM algorithm
		ADAM
	};

	/**
	 * Stores and manages dataset in memory
	 * @param INPUTS Input layer neurons
	 * @param OUTPUTS Output layer neurons
	 */
	template <int INPUTS, int OUTPUTS>
	struct dataset
	{
		struct sample
		{
			float inputs[INPUTS]{};
			float outputs[OUTPUTS]{};
		};

	private:
		sample* samples = nullptr;
		int sampleCount = 0;
		bool manageMemory = true;

	public:
		dataset()
		{
		}

		dataset(size_t reserved)
		{
			reserve(reserved);
		}

		dataset(std::initializer_list<sample> list)
		{
			reserve((int)list.size());

			int i = 0;
			for (auto elem = list.begin(); elem != list.end(); ++elem)
			{
				samples[i++] = *elem;
			}
		}

		/**
		 * Introduced for cuda
		 * @param manageMemory When enabled sample array will be deleted
		 */
		dataset(sample* samples, const int sampleCount, const bool manageMemory)
		{
			this->samples = samples;
			this->sampleCount = sampleCount;
			this->manageMemory = manageMemory;
		}

		/**
		 * Allocates new memory without clearing it (size() is updated)
		 * @param reserved New sample count
		 */
		void reserve(const int reserved)
		{
			this->~dataset();

			samples = new sample[reserved];
			sampleCount = reserved;
		}

		/**
		 * Introduced for cuda
		 * @param manageMemory When enabled sample array will be deleted
		 */
		int size() const
		{
			return sampleCount;
		}

		sample& operator[](const int i) const
		{
			return samples[i];
		}

		/**
		 * @return Highest value in dataset
		 */
		float max_value() const
		{
			float max = samples[0].inputs[0];

			for (int s = 0; s < sampleCount; ++s)
			{
				for (int i = 0; i < INPUTS; ++i)
				{
					const float& x = samples[s].inputs[i];

					if (x > max)
						max = x;
				}

				for (int i = 0; i < OUTPUTS; ++i)
				{
					const float& x = samples[s].outputs[i];

					if (x > max)
						max = x;
				}
			}

			return max;
		}

		/**
		 * @return Lowest value in dataset
		 */
		float min_value() const
		{
			float min = samples[0].inputs[0];

			for (int s = 0; s < sampleCount; ++s)
			{
				for (int i = 0; i < INPUTS; ++i)
				{
					const float& x = samples[s].inputs[i];

					if (x < min)
						min = x;
				}

				for (int i = 0; i < OUTPUTS; ++i)
				{
					const float& x = samples[s].outputs[i];

					if (x < min)
						min = x;
				}
			}

			return min;
		}

		/**
		 * Normalizes all values between max and min
		 * @param max Highest value in dataset (use this->max_value())
		 * @param min Lowest value in dataset (use this->min_value())
		 */
		void normalize(const float max, const float min)
		{
			const float range = max - min;

			for (int s = 0; s < sampleCount; ++s)
			{
				for (int i = 0; i < INPUTS; ++i)
				{
					samples[s].inputs[i] = (samples[s].inputs[i] - min) / range;
				}

				for (int i = 0; i < OUTPUTS; ++i)
				{
					samples[s].outputs[i] = (samples[s].outputs[i] - min) / range;
				}
			}
		}

		/**
		 * Randomizes order of samples
		 */
		void shuffle()
		{
			std::random_device rm_seed;
			std::shuffle(samples, samples + sampleCount - 1, std::mt19937(rm_seed()));
		}

		~dataset()
		{
			if (manageMemory)
			{
				if (samples != nullptr)
					delete[] samples;
			}
		}
	};

	struct model_parameters
	{
		bool printloss{};
		int seed{};
		std::vector<int> layout{};
		std::vector<ActivFunc> a{};
		std::string logfolder{};

		void load_parameters(const char* buffer)
		{
			const char* pointer = buffer + sizeof(uint32_t);

			memcpy(&printloss, pointer, sizeof(uint8_t));
			pointer += sizeof(uint8_t);
			memcpy(&seed, pointer, sizeof(int32_t));
			pointer += sizeof(int32_t);

			uint32_t layerCount{};
			uint32_t logLenght{};

			memcpy(&layerCount, pointer, sizeof(uint32_t));
			pointer += sizeof(uint32_t);
			memcpy(&logLenght, pointer, sizeof(uint32_t));
			pointer += sizeof(uint32_t);

			layout.resize(layerCount);
			a.resize(layerCount - 1);
			logfolder.resize(logLenght);

			std::vector<uint32_t> _layout(layerCount);

			memcpy(_layout.data(), pointer, layerCount * sizeof(uint32_t));
			pointer += layerCount * sizeof(uint32_t);
			memcpy(a.data(), pointer, a.size() * sizeof(ActivFunc));
			pointer += a.size() * sizeof(ActivFunc);
			memcpy((void*)logfolder.data(), pointer, logLenght * sizeof(char));
			// pointer += logLenght * sizeof(char);

			for (int i = 0; i < layerCount; ++i)
			{
				layout[i] = (int)_layout[i];
			}
		}

		bool load_parameters(std::ifstream& file)
		{
			if (file.is_open())
			{
				file.seekg(0, file.end);
				int N = (int)file.tellg();
				file.seekg(0, file.beg);

				std::vector<char> buffer(N);
				file.read(buffer.data(), N);
				load_parameters(buffer.data());

				return true;
			}

			return false;
		}
	};

	// Open source Neural Network library in C++
	class model
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
			* @tparam layers[l2].weights [n2 * n1Count + n1]
			*/
			float* const weights;
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
				const bool manageMemory) :
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

			layer(const int PREVIOUS, const int NEURONS, const ActivFunc func) : nCount(NEURONS), wCount(NEURONS* PREVIOUS), func(func),
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
		const bool printloss;
		const std::string logfolder;

	public:
		const int layerCount;
		layer* const layers;

		model(layer* const layers,
			const int layerCount,
			const bool printloss,
			const int seed,
			const std::string logfolder,
			const bool manageMemory) :
			layers(layers),
			layerCount(layerCount),
			seed(seed),
			printloss(printloss),
			logfolder(logfolder),
			manageMemory(manageMemory)
		{
		}

		/**
		 * Creates network where every neuron is connected to each neuron in the next layer.
		 * @param layout Starts with the input layer (Minimum 2 layers)
		 * @param a Activation function for each layer (first one will be ignored)
		 * @param seed Random weight seed (-1 generates random seed by time)
		 * @param printloss Print loss() value while training (every second, high performance consumption)
		 * @param logfolder Write log file for training progress (keep empty to disable)
		 */
		model(const std::vector<int> layout, const std::vector<ActivFunc> a, int seed = -1, const bool printloss = true, std::string logfolder = std::string())
			: manageMemory(true), seed(seed < 0 ? (unsigned int)time(NULL) : seed), printloss(printloss),
			logfolder(logfolder.size() > 0 ? ((logfolder.back() == '\\' || logfolder.back() == '/') ? logfolder : logfolder.append("/")) : logfolder),
			layerCount((int)layout.size()), layers((layer*)malloc(layerCount * sizeof(layer)))
		{
			if (layout.size() != a.size() + 1)
			{
				printf("--- error: layer and activation array not matching ---\n");
				return;
			}

			if (seed < 0)
				printf("--- Generating with Seed: %i ---\n", this->seed);

			srand(this->seed);

			new (&layers[0]) layer(0, layout[0], ActivFunc::LINEAR);
			for (int l2 = 1; l2 < layerCount; ++l2)
			{
				const int l1 = l2 - 1;

				new (&layers[l2]) layer(layout[l1], layout[l2], a[l1]);
			}
		}

		model(const model_parameters& para) : model(para.layout, para.a, para.seed, para.printloss, para.logfolder)
		{
		}

		~model()
		{
			if (manageMemory)
			{
				for (int i = 0; i < layerCount; ++i)
					layers[i].~layer();

				free(layers);
			}
		}

		/**
		 * Stores all weights and biases in one binary buffer
		 * @return size of buffer
		 */
		size_t save(std::vector<char>& buffer) const
		{
			size_t buffersize{};

			const uint8_t _printloss = (uint8_t)printloss;
			const int32_t _seed = (int32_t)seed;
			const uint32_t _layerCount = layerCount;
			const uint32_t _logLenght = logfolder.size();
			std::vector<uint32_t> layout(layerCount);
			std::vector<ActivFunc> a(layerCount - 1);
			uint32_t parameters_size = sizeof(uint8_t) + sizeof(int32_t) + (layerCount + 2) * sizeof(uint32_t) + a.size() * sizeof(ActivFunc) + _logLenght * sizeof(char);

			std::vector<size_t> nSize(layerCount);
			std::vector<size_t> wSize(layerCount);
			for (int i = 0; i < layerCount; ++i)
			{
				layout[i] = (uint32_t)layers[i].nCount;
				if (i > 0)
				{
					a[i - 1] = layers[i].func;

					nSize[i] = layers[i].nCount * sizeof(float);
					wSize[i] = layers[i].wCount * sizeof(float);
					buffersize += nSize[i] + wSize[i];
				}
			}

			buffersize += sizeof(uint32_t) + parameters_size;
			buffer.resize(buffersize);

			char* pointer = buffer.data();

			memcpy(buffer.data(), &parameters_size, sizeof(uint32_t));
			pointer += sizeof(uint32_t);

			memcpy(pointer, &_printloss, sizeof(uint8_t));
			pointer += sizeof(uint8_t);
			memcpy(pointer, &_seed, sizeof(int32_t));
			pointer += sizeof(int32_t);

			memcpy(pointer, &_layerCount, sizeof(uint32_t));
			pointer += sizeof(uint32_t);
			memcpy(pointer, &_logLenght, sizeof(uint32_t));
			pointer += sizeof(uint32_t);

			memcpy(pointer, layout.data(), layerCount * sizeof(uint32_t));
			pointer += layerCount * sizeof(uint32_t);
			memcpy(pointer, a.data(), a.size() * sizeof(ActivFunc));
			pointer += a.size() * sizeof(ActivFunc);
			memcpy(pointer, logfolder.data(), _logLenght * sizeof(char));
			pointer += _logLenght * sizeof(char);

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
		 * Stores all weights and biases in one binary file
		 * @param file use std::ios::binary
		 * @return size of buffer
		 */
		size_t save(std::ofstream& file) const
		{
			std::vector<char> buffer;
			size_t buffersize = save(buffer);

			file.write(buffer.data(), buffersize);
			if (file.fail())
				buffersize = 0;

			file.flush();

			return buffersize;
		}
		/**
		 * Loads binary network buffer
		 * @exception Currupt data can through an error
		 */
		void load(const char* buffer)
		{
			std::vector<size_t> nSize(layerCount);
			std::vector<size_t> wSize(layerCount);
			for (int i = 1; i < layerCount; ++i)
			{
				nSize[i] = layers[i].nCount * sizeof(float); // WARN float is not const size
				wSize[i] = layers[i].wCount * sizeof(float);
			}

			const char* pointer = buffer;

			uint32_t parameters_size{};

			memcpy(&parameters_size, pointer, sizeof(uint32_t));
			pointer += sizeof(uint32_t) + parameters_size;

			for (int i = 1; i < layerCount; ++i)
			{
				memcpy(layers[i].weights, pointer, wSize[i]);
				pointer += wSize[i];
				memcpy(layers[i].biases, pointer, nSize[i]);
				pointer += nSize[i];
			}
		}
		/**
		 * Loads binary network file
		 * @param file use std::ios::binary
		 * @exception Currupt files can through an error
		 * @return if successful
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
		// Copies weights and biases to a different instance
		inline void copy(model& to) const
		{
			for (int l = 1; l < layerCount; ++l)
			{
				memcpy(to.layers[l].biases, layers[l].biases, layers[l].nCount * sizeof(float));
				memcpy(to.layers[l].weights, layers[l].weights, layers[l].wCount * sizeof(float));
			}
		}
		/**
		 * Combines array of networks by averaging the values.
		 * @param with Array of networks
		 * @param count How many networks
		 */
		void combine(model* with, int count)
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
		/**
		 * Compares weights and biases
		 * @return true: equal; false: not equal
		 */
		bool equals(model& second)
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

		/**
		 * Calculates network's outputs (aka predict)
		 * @return Pointer to output layer/array
		 */
		inline float* pulse(const float* inputs, const int& N)
		{
			memcpy(layers[0].neurons, inputs, N * sizeof(float));

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

		/**
		 * sum of (1 / PARAMETERS) * (x - OPTIMAL)^2
		 * @param sample Optimal outputs (testdata)
		 * @return lower means better
		 */
		template <int INPUTS, int OUTPUTS>
		float loss(const typename dataset<INPUTS, OUTPUTS>::sample& sample)
		{
			pulse(sample.inputs, INPUTS);

			const int LASTL = layerCount - 1;

			float d = 0.0;
			for (int i = 0; i < OUTPUTS; ++i)
			{
				const float x = layers[LASTL].neurons[i] - sample.outputs[i];
				d += x * x;
			}

			return d / (float)OUTPUTS;
		}
		/**
		 * sum of (1 / PARAMETERS) * (x - OPTIMAL)^2
		 * @param data Optimal outputs (testdata)
		 * @return lower means better
		 */
		template <int INPUTS, int OUTPUTS>
		float loss(const dataset<INPUTS, OUTPUTS>& data)
		{
			if (!check(data))
			{
				printf("\tCurrupt training data!\n");
				return .0f;
			}

			float d = 0.0;
			for (int i = 0; i < data.size(); ++i)
			{
				d += loss<INPUTS, OUTPUTS>(data[i]);
			}

			return d / (float)data.size();
		}
		/**
		 * sum of (1 / PARAMETERS) * abs(x - OPTIMAL)
		 * @param sample Optimal outputs (testdata)
		 * @return lower means better
		 */
		template <int INPUTS, int OUTPUTS>
		float diff(const typename dataset<INPUTS, OUTPUTS>::sample& sample)
		{
			pulse(sample.inputs, INPUTS);

			const int LASTL = layerCount - 1;

			float d = 0.0;
			for (int i = 0; i < OUTPUTS; ++i)
			{
				d += std::abs(layers[LASTL].neurons[i] - sample.outputs[i]);
			}

			return d / (float)OUTPUTS;
		}
		/**
		 * sum of (1 / PARAMETERS) * abs(x - OPTIMAL)
		 * @param data Optimal outputs (testdata)
		 * @return lower means better
		 */
		template <int INPUTS, int OUTPUTS>
		float diff(const dataset<INPUTS, OUTPUTS>& data)
		{
			if (!check(data))
			{
				printf("\tCurrupt training data!\n");
				return .0f;
			}

			float d = 0.0;
			for (int i = 0; i < data.size(); ++i)
			{
				d += diff<INPUTS, OUTPUTS>(data[i]);
			}

			return d / (float)data.size();
		}
		/**
		 * sum of (1 / PARAMETERS) * abs(x - OPTIMAL)
		 * @param sample Optimal outputs (testdata)
		 * @return lower means better
		 */
		template <int INPUTS, int OUTPUTS>
		float diff_max(const typename dataset<INPUTS, OUTPUTS>::sample& sample)
		{
			pulse(sample.inputs, INPUTS);

			const int LASTL = layerCount - 1;

			float d = 0.0f;
			for (int i = 0; i < OUTPUTS; ++i)
			{
				const float x = std::abs(layers[LASTL].neurons[i] - sample.outputs[i]);
				if (d < x)
					d = x;
			}

			return d;
		}
		/**
		 * sum of (1 / PARAMETERS) * abs(x - OPTIMAL)
		 * @param data Optimal outputs (testdata)
		 * @return lower means better
		 */
		template <int INPUTS, int OUTPUTS>
		float diff_max(const dataset<INPUTS, OUTPUTS>& data)
		{
			if (!check(data))
			{
				printf("\tCurrupt training data!\n");
				return .0f;
			}

			float d = 0.0f;
			for (int i = 0; i < data.size(); ++i)
			{
				const float x = diff_max<INPUTS, OUTPUTS>(data[i]);
				if (d < x)
					d = x;
			}

			return d;
		}
		/**
		 * sum of (1 / PARAMETERS) * abs(x - OPTIMAL)
		 * @param sample Optimal outputs (testdata)
		 * @return lower means better
		 */
		template <int INPUTS, int OUTPUTS>
		float diff_min(const typename dataset<INPUTS, OUTPUTS>::sample& sample)
		{
			pulse(sample.inputs, INPUTS);

			const int LASTL = layerCount - 1;

			float d = std::abs(layers[LASTL].neurons[0] - sample.outputs[0]);
			for (int i = 1; i < OUTPUTS; ++i)
			{
				const float x = std::abs(layers[LASTL].neurons[i] - sample.outputs[i]);
				if (d < x)
					d = x;
			}

			return d;
		}
		/**
		 * sum of (1 / PARAMETERS) * abs(x - OPTIMAL)
		 * @param data Optimal outputs (testdata)
		 * @return lower means better
		 */
		template <int INPUTS, int OUTPUTS>
		float diff_min(const dataset<INPUTS, OUTPUTS>& data)
		{
			if (!check(data))
			{
				printf("\tCurrupt training data!\n");
				return .0f;
			}

			float d = diff_min<INPUTS, OUTPUTS>(data[0]);
			for (int i = 1; i < data.size(); ++i)
			{
				const float x = diff_min<INPUTS, OUTPUTS>(data[i]);
				if (d > x)
					d = x;
			}

			return d;
		}
		/**
		 * for each sample: does highest output matches optimal highest
		 * @param data Optimal outputs (testdata)
		 * @return percentage, higher means better
		 */
		template <int INPUTS, int OUTPUTS>
		float accruacy(const dataset<INPUTS, OUTPUTS>& data)
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
				pulse(data[i].inputs, INPUTS);

				if (std::max_element(layers[LASTL].neurons, layers[LASTL].neurons + layers[LASTL].nCount) - layers[LASTL].neurons == std::max_element(data[i].outputs, data[i].outputs + OUTPUTS) - data[i].outputs)
				{
					++a;
				}
			}

			return a / (float)data.size();
		}

		/**
		 * Prints all rating functions
		 * @return difference value "diff()"
		 */
		template <int INPUTS, int OUTPUTS>
		float evaluate(const dataset<INPUTS, OUTPUTS>& data)
		{
			float d = diff(data);

			printf("\tLoss           %f\n", loss(data));
			printf("\tDifference     %f\n", d);
			printf("\tMax Difference %f\n", d);
			printf("\tMin Difference %f\n", d);
			printf("\tAccruacy       %.1f%%\n", accruacy(data) * 100.0f);
			printf("\t-----------------------------------------------\n\n");

			return d;
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
		/**
		 * Prints all values with the loss() and the accruacy()
		 * @return difference value "diff()"
		 */
		template <int INPUTS, int OUTPUTS>
		float print(const dataset<INPUTS, OUTPUTS>& data, int sample)
		{
			pulse(data[sample].inputs, INPUTS);
			print();

			return evaluate(data);
		}
		// Prints only the output layer. pulse() should be called before
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
		/**
		 * Prints only the output layer with the loss() and the accruacy()
		 * @return difference value "diff()"
		 */
		template <int INPUTS, int OUTPUTS>
		float printo(const dataset<INPUTS, OUTPUTS>& data, const int sample)
		{
			pulse(data[sample].inputs, INPUTS);
			printo();

			return evaluate(data);
		}
		// Prints output layer in one line. pulse() should be called before
		void print_one()
		{
			const layer& last = layers[layerCount - 1];
			const int lastn = last.nCount - 1;

			printf("[");
			for (int i = 0; i < lastn; ++i)
			{
				printf("%.5f, ", last.neurons[i]);
			}
			printf("%.5f", last.neurons[lastn]);
			printf("]\n");
		}
		/**
		 * Prints output layer in one line
		 * @return difference value "diff()"
		 */
		template <int INPUTS, int OUTPUTS>
		float print_one(const dataset<INPUTS, OUTPUTS>& data, const int sample)
		{
			pulse(data[sample].inputs, INPUTS);
			print_one();

			return evaluate(data);
		}

		/**
		 * Transforms normalized output back to original scale
		 * @param max Highest value in dataset (use this->max())
		 * @param min Lowest value in dataset (use this->max())
		 * @return Denormalized output layer
		 */
		std::vector<float> denormalized(const float max, const float min) const
		{
			const layer& last = layers[layerCount - 1];
			const float range = max - min;

			std::vector<float> out(last.nCount);

			for (int i = 0; i < last.nCount; ++i)
			{
				out[i] = last.neurons[i] * range + min;
			}

			return out;
		}

		/**
		 * Trains the network with Gradient Descent to minimize the loss function (you can cancel with 'q')
		 * @param max_iterations Begin small
		 * @param traindata The large dataset
		 * @param testdata The small dataset which the network never saw before
		 * @param optimizer Search online for further information
		 * @param learningRate Lower values generate more reliable but also slower results
		 * @param momentum Can accelerate training but also produce worse results (disable with 0.0f)
		 * @param total_threads aka batch size divides training data into chunks to improve performance for large networks (not used by stochastic g.d.)
		 */
		template <int INPUTS, int OUTPUTS, int T_INPUTS, int T_OUTPUTS>
		void fit(int max_iterations, const dataset<INPUTS, OUTPUTS>& traindata, const dataset<T_INPUTS, T_OUTPUTS>& testdata,
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

				if (total_threads > traindata.size())
					total_threads = 1;

				const int batch_size = traindata.size() / total_threads;
				const int unused = traindata.size() - batch_size * total_threads;

				printf("Mini batches launching with:\n");
				printf("%i batches | ", total_threads);
				printf("%i samples per batch | ", batch_size);
				printf("%i/%i unused samples\n", unused, traindata.size());

				if (total_threads > traindata.size() / batch_size || total_threads < 0)
					total_threads = 1;

				if (optimizer == Optimizer::ADAM)
				{
					mini_batch(max_iterations, learningRate, momentum, batch_size, total_threads, traindata, testdata);
				}
			}
		}

	protected:
		template <int INPUTS, int OUTPUTS>
		inline bool check(const dataset<INPUTS, OUTPUTS>& data) const
		{
			return layers[0].nCount == INPUTS && layers[layerCount - 1].nCount == OUTPUTS;
		}

		template <int INPUTS, int OUTPUTS>
		class fitlog
		{
			model& parent;
			const int sampleCount;
			const dataset<INPUTS, OUTPUTS>& testdata;
			const int max_iterations;
			const bool printloss;

			std::unique_ptr<std::ofstream> file{};
			size_t lastLogLenght = 0;
			int lastLogAt = 0;
			std::chrono::high_resolution_clock::time_point overall;
			std::chrono::high_resolution_clock::time_point sampleTimeLast;
			std::chrono::high_resolution_clock::time_point last;

			float lastLoss[10]{};
			int lastLossIndex = 0;

			inline void resolveTime(long long seconds, int* resolved) const
			{
				resolved[0] = (int)(seconds / 3600LL);
				seconds -= 3600LL * (long long)resolved[0];
				resolved[1] = (int)(seconds / 60LL);
				seconds -= 60LL * (long long)resolved[1];
				resolved[2] = (int)(seconds);
			}

		public:
			float log(int run)
			{
				float l = -1.0f;

				auto now = std::chrono::high_resolution_clock::now();
				std::chrono::microseconds sampleTime = std::chrono::duration_cast<std::chrono::microseconds>(now - sampleTimeLast);
				if (sampleTime.count() > 1000000LL || run == max_iterations || run == 0)
				{
					sampleTimeLast = now;
					std::chrono::milliseconds elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last);
					std::chrono::seconds runtime = std::chrono::duration_cast<std::chrono::seconds>(now - overall);

					if (printloss)
						l = parent.loss(testdata);

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

					if (l > -1.0f)
						message << "loss: " << std::scientific << l;

					const int cleared = std::max(0, (int)lastLogLenght - (int)message.str().length());
					std::cout << std::string(lastLogLenght, '\b') << message.str() << std::string(cleared, ' ');
					lastLogLenght = message.str().length() + cleared;

					if (file.get() != nullptr)
						*file << progress << ',' << samplesPerSecond << ',' << runtime.count() << ',' << eta.count() << ',' << l << '\n';

					if (lastLoss[0] <= l
						&& lastLoss[1] <= l
						&& lastLoss[2] <= l
						&& lastLoss[3] <= l
						&& lastLoss[4] <= l
						&& lastLoss[5] <= l
						&& lastLoss[6] <= l
						&& lastLoss[7] <= l
						&& lastLoss[8] <= l
						&& lastLoss[9] <= l && run > 10 || kbhit() && getch() == 'q')
					{
						return 0.0f;
					}
					lastLoss[lastLossIndex++] = l;
					if (lastLossIndex == 5)
						lastLossIndex = 0;

					lastLogAt = run;
					last = std::chrono::high_resolution_clock::now();
				}

				return l;
			}

			fitlog(model& parent,
				const int& sampleCount,
				const dataset<INPUTS, OUTPUTS>& testdata,
				const int& max_iterations,
				const bool& printloss,
				const std::string& logfolder) :
				parent(parent),
				sampleCount(sampleCount),
				testdata(testdata),
				max_iterations(max_iterations),
				printloss(printloss)
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
					*file << "Progress,SamplesPer(seconds),Runtime(seconds),ETA(seconds),loss" << std::endl;
				}

				printf("\n");
				overall = std::chrono::high_resolution_clock::now();
				last = overall;
				sampleTimeLast = overall;
				lastLoss[lastLossIndex++] = 999999.0f;
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

				std::unique_ptr<model> net;
				std::vector<layer> layers;

				const float sample_share;

				batch(const model& main, const float& sample_share)
					: sample_share(sample_share)
				{
					std::vector<int> layout(main.layerCount);
					std::vector<ActivFunc> a(main.layerCount - 1);
					for (int l = 0; l < main.layerCount; ++l)
					{
						layout[l] = main.layers[l].nCount;
						if (l > 0)
							a[l - 1] = main.layers[l].func;
					}

					net.reset(new model(layout, a, main.seed, main.printloss, main.logfolder));
					main.copy(*net.get());

					layers.reserve(main.layerCount);

					layers.emplace_back(0, 0);
					for (int i = 1; i < main.layerCount; ++i)
					{
						layers.emplace_back(main.layers[i].nCount, main.layers[i].wCount);
					}
				}
			};

			std::vector<const typename dataset<INPUTS, OUTPUTS>::sample*> sample_pointers;
			std::vector<batch> batches;
			const int batch_count;
			const int layerCount;

			backprop_matrix(const float& sample_share, const int& sampleCount, const int& batch_count, const model& main, const dataset<INPUTS, OUTPUTS>& traindata)
				: batch_count(batch_count), layerCount(main.layerCount)
			{
				sample_pointers.resize(sampleCount);
				batches.reserve(batch_count);

				const typename dataset<INPUTS, OUTPUTS>::sample* sample = &traindata[0];
				for (int i = 0; i < sampleCount; ++i)
				{
					sample_pointers[i] = sample + i;
				}

				for (int i = 0; i < batch_count; ++i)
				{
					batches.emplace_back(main, sample_share);
				}
			}

			inline void apply(model& main)
			{
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
			typename backprop_matrix<INPUTS, OUTPUTS>::batch& thread, const typename dataset<INPUTS, OUTPUTS>::sample& sample)
		{
			model& net = *thread.net.get();
			const int LASTL = net.layerCount - 1;

			net.pulse(sample.inputs, INPUTS);

			{
				const int& ncount = net.layers[LASTL].nCount;
				const auto& func = net.layers[LASTL].derivative;

				for (int n2 = 0; n2 < ncount; ++n2)
				{
					thread.layers[LASTL].deltas[n2] = func(net.layers[LASTL].neurons[n2]) * 2.0f * (net.layers[LASTL].neurons[n2] - sample.outputs[n2]);
				}
			}

			for (int l2 = LASTL; l2 >= 2; --l2)
			{
				const int l1 = l2 - 1;
				const int& n1count = net.layers[l1].nCount;
				const int& n2count = net.layers[l2].nCount;
				const auto& func = net.layers[l1].derivative;

				for (int n1 = 0; n1 < n1count; ++n1)
				{
					float delta = 0;
					for (int n2 = 0; n2 < n2count; ++n2)
						delta += net.layers[l2].weights[n2 * n1count + n1] * thread.layers[l2].deltas[n2];

					thread.layers[l1].deltas[n1] = func(net.layers[l1].neurons[n1]) * delta;
				}
			}

			for (int l2 = LASTL; l2 >= 1; --l2)
			{
				const int l1 = l2 - 1;
				const int& n1count = net.layers[l1].nCount;
				const int& n2count = net.layers[l2].nCount;

				for (int n2 = 0; n2 < n2count; ++n2)
				{
					const float dn = -thread.layers[l2].deltas[n2];
					const float step = n * dn;

					thread.layers[l2].bias_gradients[n2] += (step + m * thread.layers[l2].last_gb[n2]) * thread.sample_share;
					thread.layers[l2].last_gb[n2] = step;

					const int row = n2 * n1count;
					for (int n1 = 0; n1 < n1count; ++n1)
					{
						const int windex = row + n1;
						const float dw = dn * net.layers[l1].neurons[n1];
						const float stepw = n * dw;

						thread.layers[l2].weight_gradients[windex] += (stepw + m * thread.layers[l2].last_gw[windex]) * thread.sample_share;
						thread.layers[l2].last_gw[windex] = stepw;
					}
				}
			}
		}

		template <int INPUTS, int OUTPUTS, int T_INPUTS, int T_OUTPUTS>
		void stochastic(const int& max_iterations, const float& n, const float& m, const dataset<INPUTS, OUTPUTS>& traindata, const dataset<T_INPUTS, T_OUTPUTS>& testdata)
		{
			std::vector<std::vector<float>> bias_gradients(layerCount);
			std::vector<std::vector<float>> weight_gradients(layerCount);

			for (int l = 1; l < layerCount; ++l)
			{
				bias_gradients[l].resize(layers[l].nCount);
				weight_gradients[l].resize(layers[l].wCount);
			}

			backprop_matrix<INPUTS, OUTPUTS> matrix(1.0f, traindata.size(), 1, *this, traindata);

			fitlog<T_INPUTS, T_OUTPUTS> logger(*this, traindata.size(), testdata, max_iterations, printloss, logfolder);

			for (int run = 1; run <= max_iterations; ++run)
			{
				for (int i = 0; i < traindata.size(); ++i)
				{
					backprop<INPUTS, OUTPUTS>(n, m, matrix.batches[0], traindata[i]);
					matrix.apply(*this);
				}

				if (logger.log(run) == 0.0f)
					break;
			}

			printf("\n\n");
		}

		template <int INPUTS, int OUTPUTS, int T_INPUTS, int T_OUTPUTS>
		void mini_batch(const int& max_iterations, const float& n, const float& m, const int& batch_size, const int& total_threads, const dataset<INPUTS, OUTPUTS>& traindata, const dataset<T_INPUTS, T_OUTPUTS>& testdata)
		{
			const int sampleCount = batch_size * total_threads;

			std::vector<std::thread> threads(total_threads);

			backprop_matrix<INPUTS, OUTPUTS> matrix(1.0f / (float)sampleCount, sampleCount, total_threads, *this, traindata);

			fitlog<T_INPUTS, T_OUTPUTS> logger(*this, traindata.size(), testdata, max_iterations, printloss, logfolder);

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

				if (logger.log(run) == 0.0f)
					break;
			}

			printf("\n\n");
		}
	};
}

#endif