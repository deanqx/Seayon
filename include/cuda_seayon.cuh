#pragma once 
// #include "cuda_runtime.h"
// #include "device_launch_parameters.h"
#include <vector>
#include "seayon.hpp"

// __device__
float device_Sigmoid(const float& z)
{
	return 1.0f / (1.0f + exp(-z));
}
// __device__
float device_dSigmoid(const float& a)
{
	return a * (1.0f - a);
}
// __device__
float device_Tanh(const float& z)
{
	float x0 = exp(z);
	float x1 = exp(-z);
	return (x0 - x1) / (x0 + x1);
}
// __device__
float device_dTanh(const float& a)
{
	float t = Tanh(a);
	return 1 - t * t;
}
// __device__
float device_ReLu(const float& z)
{
	return (z < 0.0f ? 0.0f : z);
}
// __device__
float device_dReLu(const float& a)
{
	return (a < 0.0f ? 0.0f : 1.0f);
}
// __device__
float device_LeakyReLu(const float& z)
{
	float x = 0.1f * z;
	return (x > z ? x : z);
}
// __device__
float device_dLeakyReLu(const float& a)
{
	return (a > 0.0f ? 1.0f : 0.01f);
}

enum class ParallelOptimizer
{
	MINI_BATCH,
	ADAM
};

class cuda_seayon: public seayon
{
private:
	using seayon::manageMemory;
	using seayon::printEnabled;
	using seayon::printcost;
	using seayon::seed;
	using seayon::logfolder;
	using seayon::layers;
	using seayon::layerCount;
public:
	using seayon::seayon;
	using seayon::size;
	using seayon::operator[];
	using seayon::save;
	using seayon::load;
	using seayon::copy;
	using seayon::combine;
	using seayon::equals;
	using seayon::pulse;
	using seayon::print;
	using seayon::printo;
	using seayon::cost;
	using seayon::accruacy;
private:
	using seayon::resolveTime;
	using seayon::log;
	using seayon::init_log;

public:
	template <int INPUTS, int OUTPUTS>
	struct memory_manager
	{
		struct device;
		struct host
		{
			struct batch
			{
				struct layer
				{
					std::vector<float> bias_gradients;
					std::vector<float> weight_gradients;

					layer(const int nCount, const int wCount)
					{
						bias_gradients.resize(nCount);
						weight_gradients.resize(wCount);
					}
				};

				std::vector<layer> layers;

				batch(const cuda_seayon& main)
				{
					layers.reserve(main.layerCount);

					for (size_t i = 0; i < main.layerCount; ++i)
					{
						layers.emplace_back(main.layers[i].nCount, main.layers[i].wCount);
					}
				}
			};

			std::unique_ptr<device> linker_device;

			std::vector<batch> batches;
			std::vector<const typename trainingdata<INPUTS, OUTPUTS>::sample*> device_sample_pointers;

			const int batch_count;
			const float n;
			const float m;

			host(const int batch_count, const float n, const float m, const int sampleCount, const cuda_seayon& main): batch_count(batch_count), n(n), m(m)
			{
				batches.reserve(batch_count);
				device_sample_pointers.resize(sampleCount);

				for (size_t i = 0; i < batch_count; ++i)
				{
					batches.emplace_back(main);
				}
			}
		};

		struct device
		{
			struct batch
			{
				struct layer
				{
					float* deltas;
					float* last_gb;
					float* last_gw;
					float* bias_gradients;
					float* weight_gradients;

					const size_t nSize;
					const size_t wSize;

					layer(const int nCount, const int wCount): nSize(nCount * sizeof(float)), wSize(wCount * sizeof(float))
					{
						deltas = (float*)malloc(nSize);
						last_gb = (float*)malloc(nSize);
						last_gw = (float*)malloc(wSize);
						bias_gradients = (float*)malloc(nSize);
						weight_gradients = (float*)malloc(wSize);
					}

					void reset()
					{
						memset(deltas, 0, nSize);
						memset(last_gb, 0, nSize);
						memset(last_gw, 0, wSize);
						memset(bias_gradients, 0, nSize);
						memset(weight_gradients, 0, wSize);
					}
				};

				layer* layers;
				std::vector<layer> linker_layers;
				cuda_seayon* net;
				std::unique_ptr<cuda_seayon> linker_net;
				std::vector<seayon::layer> linker_seayon_layers;

				batch(const cuda_seayon& main)
				{
					layers = (layer*)malloc(main.layerCount * sizeof(layer));
					linker_layers.reserve(main.layerCount);
					net = (cuda_seayon*)malloc(sizeof(cuda_seayon));
					linker_seayon_layers.reserve(main.layerCount);

					for (int i = 0; i < main.layerCount; ++i)
					{
						linker_layers.emplace_back(main.layers[i].nCount, main.layers[i].wCount);

						float* neurons;
						float* biases;
						float* weights;

						neurons = (float*)malloc(main.layers[i].nCount * sizeof(float));
						// cudaMalloc(&neurons, layers[i].nCount * sizeof(float));
						if (i > 0)
						{
							biases = (float*)malloc(main.layers[i].nCount * sizeof(float));
							weights = (float*)malloc(main.layers[i].wCount * sizeof(float));
							// cudaMalloc(&biases, layers[i].nCount * sizeof(float));
							// cudaMalloc(&weights, layers[i].wCount * sizeof(float));
						}

						const ActivFunc& func = main.layers[i].func;
						float (*activation)(const float& z);
						float (*derivative)(const float& a);
						if (func == ActivFunc::SIGMOID)
						{
							activation = device_Sigmoid;
							derivative = device_dSigmoid;
						}
						else if (func == ActivFunc::TANH)
						{
							activation = device_Tanh;
							derivative = device_dTanh;
						}
						else if (func == ActivFunc::RELU)
						{
							activation = device_ReLu;
							derivative = device_dReLu;
						}
						else if (func == ActivFunc::LEAKYRELU)
						{
							activation = device_LeakyReLu;
							derivative = device_dLeakyReLu;
						}

						linker_seayon_layers.emplace_back(func, activation, derivative, neurons, biases, weights, main.layers[i].nCount, main.layers[i].wCount, false);
					}

					memcpy(layers, linker_layers.data(), main.layerCount * sizeof(layer));

					seayon::layer* seayon_layers;
					seayon_layers = (seayon::layer*)malloc(main.layerCount * sizeof(seayon::layer));
					memcpy(seayon_layers, linker_seayon_layers.data(), main.layerCount * sizeof(seayon::layer));

					linker_net.reset(new cuda_seayon(seayon_layers, main.layerCount, main.printEnabled, main.printcost, main.seed, main.logfolder, false));
					memcpy(net, linker_net.get(), sizeof(cuda_seayon));
					reset(main);
				}

				void reset(const cuda_seayon& main)
				{
					for (int i = 0; i < main.layerCount; ++i)
					{
						linker_layers[i].reset();

						memset(linker_seayon_layers[i].neurons, 0, main.layers[i].nCount * sizeof(float));
						// cudaMemset(linker_seayon_layers[i].neurons, 0, main.layers[i].nCount * sizeof(float));
						if (i > 0)
						{
							memcpy(linker_seayon_layers[i].weights, main.layers[i].weights, main.layers[i].wCount * sizeof(float));
							memcpy(linker_seayon_layers[i].biases, main.layers[i].biases, main.layers[i].nCount * sizeof(float));
							// cudaMemcpy(linker_seayon_layers[i].weights, main.layers[i].weights, main.layers[i].wCount * sizeof(float), cudaMemcpyHostToDevice);
							// cudaMemcpy(linker_seayon_layers[i].biases, main.layers[i].biases, main.layers[i].nCount * sizeof(float), cudaMemcpyHostToDevice);
						}
					}
				}
			};

			batch* batches;
			std::vector<batch> linker_batches;

			trainingdata<INPUTS, OUTPUTS>* traindata;
			std::unique_ptr<trainingdata<INPUTS, OUTPUTS>> linker_traindata;
			const typename trainingdata<INPUTS, OUTPUTS>::sample** sample_pointers;

			const int layerCount;
			const int batch_count;
			const int batch_size;
			const float n;
			const float m;

			device(const int& batch_count, const int& batch_size, const float& n, const float& m, const cuda_seayon& main, const trainingdata<INPUTS, OUTPUTS>& host_traindata)
				: layerCount(main.layerCount), batch_count(batch_count), batch_size(batch_size), n(n), m(m)
			{
				const int used_size = batch_count * batch_size;

				batches = (batch*)malloc(batch_count * sizeof(batch));
				linker_batches.reserve(batch_count);
				traindata = (trainingdata<INPUTS, OUTPUTS>*)malloc(sizeof(trainingdata<INPUTS, OUTPUTS>));
				sample_pointers = (const typename trainingdata<INPUTS, OUTPUTS>::sample**)malloc(used_size * sizeof(const typename trainingdata<INPUTS, OUTPUTS>::sample*));

				for (size_t i = 0; i < batch_count; ++i)
				{
					linker_batches.emplace_back(main);
				}

				memcpy(batches, linker_batches.data(), batch_count * sizeof(batch));

				typename trainingdata<INPUTS, OUTPUTS>::sample* samples = (typename trainingdata<INPUTS, OUTPUTS>::sample*)malloc(used_size * sizeof(typename trainingdata<INPUTS, OUTPUTS>::sample));;
				linker_traindata.reset(new trainingdata<INPUTS, OUTPUTS>(samples, used_size, false));

				memcpy(samples, &host_traindata[0], used_size * sizeof(typename trainingdata<INPUTS, OUTPUTS>::sample));
				memcpy(traindata, linker_traindata.get(), sizeof(trainingdata<INPUTS, OUTPUTS>));
			}

			void reset(const cuda_seayon& main)
			{
				for (size_t i = 0; i < batch_count; ++i)
				{
					linker_batches[i].reset(main);
				}
			}
		};

		host on_host;
		device* on_device;

		memory_manager(const cuda_seayon& main, const trainingdata<INPUTS, OUTPUTS>& traindata, const int& batch_count, const int& batch_size, const float& n, const float& m)
			: on_host(batch_count, n, m, traindata.size(), main)
		{
			on_device = (device*)malloc(sizeof(device));
			on_host.linker_device.reset(new device(batch_count, batch_size, n, m, main, traindata));

			const typename trainingdata<INPUTS, OUTPUTS>::sample* device_samples = &(*on_host.linker_device->linker_traindata.get())[0]; // TEMP -> &(*on_host.linker_device->linker_traindata)[0]
			for (size_t i = 0; i < traindata.size(); ++i)
			{
				on_host.device_sample_pointers[i] = device_samples + i;
			}

			memcpy(on_host.linker_device->sample_pointers, on_host.device_sample_pointers.data(), traindata.size() * sizeof(const typename trainingdata<INPUTS, OUTPUTS>::sample*));
			memcpy(on_device, on_host.linker_device.get(), sizeof(device));
		}


		void sync()
		{
			for (int b = 0; b < on_host.batch_count; ++b)
			{
				for (int l = 0; l < on_host.linker_device->layerCount; ++l)
				{
					// printf("bias_gradients[0]: %f | size: %llu\n", on_host.linker_device->linker_batches[b].linker_layers[l].bias_gradients[0], on_host.batches[b].layers[l].bias_gradients.size());
					// if (l > 0)
					// 	printf("weight_gradients[0]: %f | size: %llu\n", on_host.linker_device->linker_batches[b].linker_layers[l].weight_gradients[0], on_host.batches[b].layers[l].weight_gradients.size());

					memcpy(on_host.batches[b].layers[l].bias_gradients.data(),
						on_host.linker_device->linker_batches[b].linker_layers[l].bias_gradients,
						on_host.batches[b].layers[l].bias_gradients.size() * sizeof(float));

					memcpy(on_host.batches[b].layers[l].weight_gradients.data(),
						on_host.linker_device->linker_batches[b].linker_layers[l].weight_gradients,
						on_host.batches[b].layers[l].weight_gradients.size() * sizeof(float));
				}
			}
		}

		void reset(const cuda_seayon& main)
		{
			on_host.linker_device->reset(main);
		}

		void shuffle()
		{
			std::random_device rm_seed;
			std::shuffle(on_host.device_sample_pointers.begin(), on_host.device_sample_pointers.end(), std::mt19937(rm_seed()));

			memcpy(on_host.linker_device->sample_pointers, on_host.device_sample_pointers.data(), on_host.device_sample_pointers.size() * sizeof(typename trainingdata<INPUTS, OUTPUTS>::sample*));
		}

		~memory_manager()
		{
		}
	};

private:
	template <int INPUTS, int OUTPUTS, int T_INPUTS, int T_OUTPUTS>
	void mini_batch(const int& max_iterations, const float& n, const float& m, const int& batch_size, const trainingdata<INPUTS, OUTPUTS>& traindata, const trainingdata<T_INPUTS, T_OUTPUTS>& testdata);

public:
	template <int INPUTS, int OUTPUTS, int T_INPUTS, int T_OUTPUTS>
	void fit(int max_iterations, const trainingdata<INPUTS, OUTPUTS>& traindata, const trainingdata<T_INPUTS, T_OUTPUTS>& testdata, ParallelOptimizer optimizer = ParallelOptimizer::MINI_BATCH, float learningRate = 0.03f, float momentum = 0.1f, int batch_size = 50)
	{
		if (!check(traindata) || !check(testdata))
		{
			if (printEnabled)
				printf("\tCurrupt training data!\n");
			return;
		}

		if (optimizer == ParallelOptimizer::MINI_BATCH)
		{
			if (batch_size > traindata.size())
				batch_size = traindata.size();

			mini_batch(max_iterations, learningRate, momentum, batch_size, traindata, testdata);
		}
		else if (optimizer == ParallelOptimizer::ADAM)
		{
		}
	}
};

// __host__ __device__
template <int INPUTS, int OUTPUTS>
void cuda_gradient_descent(const int& b, typename cuda_seayon::memory_manager<INPUTS, OUTPUTS>::device& mm)
{
	const int LASTL = mm.layerCount - 1;
	typename cuda_seayon::memory_manager<INPUTS, OUTPUTS>::device::batch& batch = mm.batches[b];
	cuda_seayon& net = *batch.net;
	const float n = mm.n;
	const float m = mm.m;

	for (int i = 0; i < mm.batch_size; ++i)
	{
		const typename trainingdata<INPUTS, OUTPUTS>::sample& sample = *mm.sample_pointers[b * mm.batch_size + i];
		net.template pulse<INPUTS, OUTPUTS>(sample);

		{
			const int l1 = LASTL - 1;
			const int& n1count = net[l1].nCount;
			const int& n2count = net[LASTL].nCount;

			for (int n2 = 0; n2 < n2count; ++n2)
			{
				const int row = n2 * n1count;

				const float delta = net[LASTL].derivative(net[LASTL].neurons[n2]) * 2.0f * (net[LASTL].neurons[n2] - sample.outputs[n2]);
				const float gradient = -delta * n;

				batch.layers[LASTL].deltas[n2] = delta;
				batch.layers[LASTL].bias_gradients[n2] += gradient;

				for (int n1 = 0; n1 < n1count; ++n1)
					batch.layers[LASTL].weight_gradients[row + n1] += gradient * net[l1].neurons[n1];
			}
		}

		for (int l2 = LASTL; l2 >= 2; --l2)
		{
			const int l1 = l2 - 1;
			const int& n1count = net[l1].nCount;
			const int& n2count = net[l2].nCount;
			const auto& deri = net[l1].derivative;

			for (int n1 = 0; n1 < n1count; ++n1)
			{
				float error = 0;
				for (int n2 = 0; n2 < n2count; ++n2)
					error += batch.layers[l2].deltas[n2] * net[l2].weights[n2 * n1count + n1];

				batch.layers[l1].deltas[n1] = deri(net[l1].neurons[n1]) * error;
			}
		}

		// PERF Try to combine loops
		for (int l2 = LASTL; l2 >= 1; --l2)
		{
			const int l1 = l2 - 1;
			const int& n1count = net[l1].nCount;
			const int& n2count = net[l2].nCount;

			for (int n2 = 0; n2 < n2count; ++n2)
			{
				const float d = -batch.layers[l2].deltas[n2];
				const float gradient = d * n;

				batch.layers[l2].bias_gradients[n2] += gradient + m * batch.layers[l2].last_gb[n2];
				batch.layers[l2].last_gb[n2] = d;

				const int row = n2 * n1count;
				for (int n1 = 0; n1 < n1count; ++n1)
				{
					const int windex = row + n1;
					const float gw = gradient * net[l1].neurons[n1];

					batch.layers[l2].weight_gradients[windex] += gw + m * batch.layers[l2].last_gw[windex];
					batch.layers[l2].last_gw[windex] = gw;
				}
			}
		}
	}
}

template <int layerCount, int INPUTS, int OUTPUTS>
__global__ void kernel(typename cuda_seayon::memory_manager<INPUTS, OUTPUTS>::device* on_device)
{
	const int b = blockIdx.x * blockDim.x + threadIdx.x;

	cuda_gradient_descent<INPUTS, OUTPUTS>(b, *on_device);
}

template <int INPUTS, int OUTPUTS, int T_INPUTS, int T_OUTPUTS>
void cuda_seayon::mini_batch(const int& max_iterations, const float& n, const float& m, const int& batch_size, const trainingdata<INPUTS, OUTPUTS>& traindata, const trainingdata<T_INPUTS, T_OUTPUTS>& testdata)
{
	const int LASTL = layerCount - 1;
	const int block_count = traindata.size() / batch_size / 512 + 1;      // Optimal: power of 2
	const int thread_count = traindata.size() / batch_size / block_count; // Optimal: multiple of 32, range 128 and 512
	const int batch_count = block_count * thread_count;
	printf("blocks: %i | threads: %i | batch_count: %i | batch_size: %i | used: %i\n", block_count, thread_count, batch_count, batch_size, batch_count * batch_size);

	std::unique_ptr<std::ofstream> logfile{};

	if (printEnabled)
	{
		float c = -1.0f;
		if (printcost)
			c = cost(testdata);

		init_log(c, max_iterations, traindata.size(), logfile);
	}

	memory_manager<INPUTS, OUTPUTS> mm(*this, traindata, batch_count, batch_size, n, m);

	auto overall = std::chrono::high_resolution_clock::now();
	auto last = std::chrono::high_resolution_clock::now();
	auto sampleTimeLast = std::chrono::high_resolution_clock::now();

	for (int run = 1; run <= max_iterations; ++run)
	{
		// kernel<INPUTS, OUTPUTS> << <block_count, thread_count >> > (mm.on_device);

		for (int b = 0; b < batch_count; ++b)
		{
			cuda_gradient_descent<INPUTS, OUTPUTS>(b, *mm.on_device);
		}

		// cudaDeviceSynchronize();
		mm.sync();

		for (int b = 0; b < batch_count; ++b)
		{
			for (int l2 = LASTL; l2 >= 1; --l2)
			{
				const int l1 = l2 - 1;
				const int& n1count = layers[l1].nCount;
				const int& n2count = layers[l2].nCount;

				for (int n2 = 0; n2 < n2count; ++n2)
				{
					layers[l2].biases[n2] += mm.on_host.batches[b].layers[l2].bias_gradients[n2];

					const int row = n2 * n1count;
					for (int n1 = 0; n1 < n1count; ++n1)
					{
						const int windex = row + n1;
						layers[l2].weights[windex] += mm.on_host.batches[b].layers[l2].weight_gradients[windex];
					}
				}
			}
		}

		mm.reset(*this);
		mm.shuffle();

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

			log(logfile, run, max_iterations, traindata.size(), runtime.count(), elapsed, sampleTime, c);

			sampleTimeLast = std::chrono::high_resolution_clock::now();
		}
	}


	if (printEnabled)
	{
		std::cout << std::endl
			<< std::endl;
	}
}