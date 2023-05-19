#pragma once 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <memory>
#include "seayon.hpp"

__device__ float cuda_Sigmoid(const float& z)
{
	return 1.0f / (1.0f + exp(-z));
}
__device__ float cuda_dSigmoid(const float& a)
{
	return a * (1.0f - a);
}
__device__ float cuda_Tanh(const float& z)
{
	float x0 = exp(z);
	float x1 = exp(-z);
	return (x0 - x1) / (x0 + x1);
}
__device__ float cuda_dTanh(const float& a)
{
	float t = tanh(a);
	return 1 - t * t;
}
__device__ float cuda_ReLu(const float& z)
{
	return (z < 0.0f ? 0.0f : z);
}
__device__ float cuda_dReLu(const float& a)
{
	return (a < 0.0f ? 0.0f : 1.0f);
}
__device__ float cuda_LeakyReLu(const float& z)
{
	float x = 0.1f * z;
	return (x > z ? x : z);
}
__device__ float cuda_dLeakyReLu(const float& a)
{
	return (a > 0.0f ? 1.0f : 0.01f);
}

typedef float(*ActivFunc_t)(const float&);

__device__ ActivFunc_t d_Sigmoid = cuda_Sigmoid;
__device__ ActivFunc_t d_dSigmoid = cuda_dSigmoid;
__device__ ActivFunc_t d_Tanh = cuda_Tanh;
__device__ ActivFunc_t d_dTanh = cuda_dTanh;
__device__ ActivFunc_t d_ReLu = cuda_ReLu;
__device__ ActivFunc_t d_dReLu = cuda_dReLu;
__device__ ActivFunc_t d_LeakyReLu = cuda_LeakyReLu;
__device__ ActivFunc_t d_dLeakyReLu = cuda_dLeakyReLu;
ActivFunc_t h_Sigmoid;
ActivFunc_t h_dSigmoid;
ActivFunc_t h_Tanh;
ActivFunc_t h_dTanh;
ActivFunc_t h_ReLu;
ActivFunc_t h_dReLu;
ActivFunc_t h_LeakyReLu;
ActivFunc_t h_dLeakyReLu;

enum class ParallelOptimizer
{
	MINI_BATCH,
	ADAM
};

class cuda_seayon : public seayon::model
{
private:
	using model::manageMemory;
	using model::printloss;
	using model::seed;
	using model::logfolder;
public:
	using model::layers;
	using model::layers.size();

	using model::model;
	using model::save;
	using model::load;
	using model::copy;
	using model::combine;
	using model::equals;
	using model::pulse;
	using model::print;
	using model::printo;
	using model::loss;
	using model::accruacy;
private:
	using model::fitlog;

public:
	cuda_seayon(const std::vector<int> layout, const std::vector<ActivFunc> a, int seed = -1, const bool printloss = true, std::string logfolder = std::string())
		: model(layout, a, seed, printloss, logfolder)
	{
		cudaMemcpyFromSymbol(&h_Sigmoid, d_Sigmoid, sizeof(ActivFunc_t));
		cudaMemcpyFromSymbol(&h_dSigmoid, d_dSigmoid, sizeof(ActivFunc_t));
		cudaMemcpyFromSymbol(&h_Tanh, d_Tanh, sizeof(ActivFunc_t));
		cudaMemcpyFromSymbol(&h_dTanh, d_dTanh, sizeof(ActivFunc_t));
		cudaMemcpyFromSymbol(&h_ReLu, d_ReLu, sizeof(ActivFunc_t));
		cudaMemcpyFromSymbol(&h_dReLu, d_dReLu, sizeof(ActivFunc_t));
		cudaMemcpyFromSymbol(&h_LeakyReLu, d_LeakyReLu, sizeof(ActivFunc_t));
		cudaMemcpyFromSymbol(&h_dLeakyReLu, d_dLeakyReLu, sizeof(ActivFunc_t));
	}

	template <int INPUTS, int OUTPUTS>
	struct backprop_matrix
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
					layers.reserve(main.layers.size());

					for (size_t i = 0; i < main.layers.size(); ++i)
					{
						layers.emplace_back(main.layers[i].nCount, main.layers[i].wCount);
					}
				}
			};

			std::unique_ptr<device> linker_device;

			std::vector<batch> batches;
			std::vector<const typename dataset<INPUTS, OUTPUTS>::sample*> device_sample_pointers;

			const int batch_count;
			const float n;
			const float m;

			host(const int batch_count, const float n, const float m, const int samples.size(), const cuda_seayon& main) : batch_count(batch_count), n(n), m(m)
			{
				batches.reserve(batch_count);
				device_sample_pointers.resize(samples.size());

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

					layer(const int nCount, const int wCount) : nSize(nCount * sizeof(float)), wSize(wCount * sizeof(float))
					{
						cudaMalloc(&deltas, nSize);
						cudaMalloc(&last_gb, nSize);
						cudaMalloc(&last_gw, wSize);
						cudaMalloc(&bias_gradients, nSize);
						cudaMalloc(&weight_gradients, wSize);
					}

					void reset()
					{
						cudaMemset(deltas, 0, nSize);
						cudaMemset(last_gb, 0, nSize);
						cudaMemset(last_gw, 0, wSize);
						cudaMemset(bias_gradients, 0, nSize);
						cudaMemset(weight_gradients, 0, wSize);
					}

					~layer()
					{
						cudaFree(deltas);
						cudaFree(last_gb);
						cudaFree(last_gw);
						cudaFree(bias_gradients);
						cudaFree(weight_gradients);
					}
				};

				layer* layers;
				cuda_seayon* net;
				std::vector<layer> linker_layers;
				std::unique_ptr<cuda_seayon> linker_net;
				std::vector<seayon::layer> linker_seayon_layers;

				const float sample_share;

				batch(const batch& b) : sample_share(0.0f)
				{
					printf("\n\n--- cuda error ---\n\n");
				}

				batch(const cuda_seayon& main, const float& sample_share)
					: sample_share(sample_share)
				{
					cudaMalloc(&layers, main.layers.size() * sizeof(layer));
					cudaMalloc(&net, sizeof(cuda_seayon));
					linker_layers.reserve(main.layers.size());
					linker_seayon_layers.reserve(main.layers.size());

					for (int i = 0; i < main.layers.size(); ++i)
					{
						linker_layers.emplace_back(main.layers[i].nCount, main.layers[i].wCount);

						float* neurons;
						float* biases;
						float* weights;

						cudaMalloc(&neurons, main.layers[i].nCount * sizeof(float));
						if (i > 0)
						{
							cudaMalloc(&biases, main.layers[i].nCount * sizeof(float));
							cudaMalloc(&weights, main.layers[i].wCount * sizeof(float));
						}

						const ActivFunc& func = main.layers[i].func;
						ActivFunc_t activation;
						ActivFunc_t derivative;
						if (func == ActivFunc::SIGMOID)
						{
							activation = h_Sigmoid;
							derivative = h_dSigmoid;
						}
						else if (func == ActivFunc::TANH)
						{
							activation = h_Tanh;
							derivative = h_dTanh;
						}
						else if (func == ActivFunc::RELU)
						{
							activation = h_ReLu;
							derivative = h_dReLu;
						}
						else if (func == ActivFunc::LEAKYRELU)
						{
							activation = h_LeakyReLu;
							derivative = h_dLeakyReLu;
						}

						linker_seayon_layers.emplace_back(func, activation, derivative, neurons, biases, weights, main.layers[i].nCount, main.layers[i].wCount, false);
					}

					cudaMemcpy(layers, linker_layers.data(), main.layers.size() * sizeof(layer), cudaMemcpyHostToDevice);

					seayon::layer* seayon_layers;
					cudaMalloc(&seayon_layers, main.layers.size() * sizeof(seayon::layer));
					cudaMemcpy(seayon_layers, linker_seayon_layers.data(), main.layers.size() * sizeof(seayon::layer), cudaMemcpyHostToDevice);
					linker_net.reset(new cuda_seayon(seayon_layers, main.layers.size(), main.printloss, main.seed, main.logfolder, false));

					cudaMemcpy(net, linker_net.get(), sizeof(cuda_seayon), cudaMemcpyHostToDevice);
					reset(main);
				}

				void reset(const cuda_seayon& main)
				{
					for (int i = 0; i < main.layers.size(); ++i)
					{
						linker_layers[i].reset();

						cudaMemset(linker_seayon_layers[i].neurons, 0, main.layers[i].nCount * sizeof(float));
						if (i > 0)
						{
							cudaMemcpy(linker_seayon_layers[i].weights, main.layers[i].weights, main.layers[i].wCount * sizeof(float), cudaMemcpyHostToDevice);
							cudaMemcpy(linker_seayon_layers[i].biases, main.layers[i].biases, main.layers[i].nCount * sizeof(float), cudaMemcpyHostToDevice);
						}
					}
				}

				~batch()
				{
					cudaFree(layers);
					cudaFree(net);
					for (int i = 0; i < linker_net->layers.size(); ++i)
					{
						cudaFree(linker_seayon_layers[i].neurons);
						cudaFree(linker_seayon_layers[i].biases);
						cudaFree(linker_seayon_layers[i].weights);
					}
					cudaFree(linker_net->layers);
				}
			};

			batch* batches;
			dataset<INPUTS, OUTPUTS>* traindata;
			const typename dataset<INPUTS, OUTPUTS>::sample** sample_pointers;

			std::vector<batch> linker_batches;
			std::unique_ptr<dataset<INPUTS, OUTPUTS>> linker_traindata;

			const int layers.size();
			const int batch_count;
			const int batch_size;
			const float n;
			const float m;

			device(const int& batch_count, const int& batch_size, const float& n, const float& m, const cuda_seayon& main, const dataset<INPUTS, OUTPUTS>& host_traindata)
				: layers.size()(main.layers.size()), batch_count(batch_count), batch_size(batch_size), n(n), m(m)
			{
				const int used_size = batch_count * batch_size;
				const float sample_share = 1.0f / (float)used_size;

				cudaMalloc(&batches, batch_count * sizeof(batch));
				cudaMalloc(&traindata, sizeof(dataset<INPUTS, OUTPUTS>));
				cudaMalloc(&sample_pointers, host_traindata.samples.size() * sizeof(const typename dataset<INPUTS, OUTPUTS>::sample*));
				linker_batches.reserve(batch_count);

				for (size_t i = 0; i < batch_count; ++i)
				{
					linker_batches.emplace_back(main, sample_share);
				}

				cudaMemcpy(batches, linker_batches.data(), batch_count * sizeof(batch), cudaMemcpyHostToDevice);

				typename dataset<INPUTS, OUTPUTS>::sample* samples;
				cudaMalloc(&samples, used_size * sizeof(typename dataset<INPUTS, OUTPUTS>::sample));;
				cudaMemcpy(samples, &host_traindata[0], used_size * sizeof(typename dataset<INPUTS, OUTPUTS>::sample), cudaMemcpyHostToDevice);
				linker_traindata.reset(new dataset<INPUTS, OUTPUTS>(samples, used_size, false));

				cudaMemcpy(traindata, linker_traindata.get(), sizeof(dataset<INPUTS, OUTPUTS>), cudaMemcpyHostToDevice);
			}

			void reset(const cuda_seayon& main)
			{
				for (size_t i = 0; i < batch_count; ++i)
				{
					linker_batches[i].reset(main);
				}
			}

			~device()
			{
				cudaFree(batches);
				cudaFree(traindata);
				cudaFree(sample_pointers);
				cudaFree(&((*linker_traindata.get())[0]));
			}
		};

		host on_host;
		device* on_device;

		backprop_matrix(const cuda_seayon& main, const dataset<INPUTS, OUTPUTS>& traindata, const int& batch_count, const int& batch_size, const float& n, const float& m)
			: on_host(batch_count, n, m, traindata.samples.size(), main)
		{
			cudaMalloc(&on_device, sizeof(device));
			on_host.linker_device.reset(new device(batch_count, batch_size, n, m, main, traindata));

			const typename dataset<INPUTS, OUTPUTS>::sample* device_samples = &(*on_host.linker_device->linker_traindata)[0];
			for (int i = 0; i < traindata.samples.size(); ++i)
			{
				on_host.device_sample_pointers[i] = device_samples + i;
			}

			cudaMemcpy(on_host.linker_device->sample_pointers, on_host.device_sample_pointers.data(), traindata.samples.size() * sizeof(typename dataset<INPUTS, OUTPUTS>::sample*), cudaMemcpyHostToDevice);
			cudaMemcpy(on_device, on_host.linker_device.get(), sizeof(device), cudaMemcpyHostToDevice);
		}


		void sync()
		{
			for (int b = 0; b < on_host.batch_count; ++b)
			{
				for (int l = 0; l < on_host.linker_device->layers.size(); ++l)
				{
					cudaMemcpy(on_host.batches[b].layers[l].bias_gradients.data(),
						on_host.linker_device->linker_batches[b].linker_layers[l].bias_gradients,
						on_host.batches[b].layers[l].bias_gradients.size() * sizeof(float),
						cudaMemcpyDeviceToHost);

					cudaMemcpy(on_host.batches[b].layers[l].weight_gradients.data(),
						on_host.linker_device->linker_batches[b].linker_layers[l].weight_gradients,
						on_host.batches[b].layers[l].weight_gradients.size() * sizeof(float),
						cudaMemcpyDeviceToHost);
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

			cudaMemcpy(on_host.linker_device->sample_pointers,
				on_host.device_sample_pointers.data(),
				on_host.device_sample_pointers.size() * sizeof(typename dataset<INPUTS, OUTPUTS>::sample*),
				cudaMemcpyHostToDevice);
		}

		~backprop_matrix()
		{
			cudaFree(on_device);
		}
	};

private:
	template <int INPUTS, int OUTPUTS, int T_INPUTS, int T_OUTPUTS>
	void mini_batch(const int& max_iterations, const float& n, const float& m, const int& batch_size, int total_threads, const dataset<INPUTS, OUTPUTS>& traindata, const dataset<T_INPUTS, T_OUTPUTS>& testdata);

public:
	template <int INPUTS, int OUTPUTS, int T_INPUTS, int T_OUTPUTS>
	void fit(int max_iterations, const dataset<INPUTS, OUTPUTS>& traindata, const dataset<T_INPUTS, T_OUTPUTS>& testdata, ParallelOptimizer optimizer = ParallelOptimizer::MINI_BATCH, float learningRate = 0.03f, float momentum = 0.1f, int total_threads = 128)
	{
		if (!check(traindata) || !check(testdata))
		{
			printf("\tCurrupt training data!\n");
			return;
		}

		if (learningRate <= 0.0f)
			learningRate = 0.0001f;

		if (momentum <= 0.0f)
			momentum = 0.0001f;

		const int batch_size = traindata.samples.size() / total_threads;
		const int unused = traindata.samples.size() - batch_size * total_threads;

		if (total_threads > traindata.samples.size() / batch_size || total_threads < 0)
			total_threads = 1;

		if (optimizer == ParallelOptimizer::MINI_BATCH)
		{
			mini_batch(max_iterations, learningRate, momentum, batch_size, total_threads, traindata, testdata);
		}
		else if (optimizer == ParallelOptimizer::ADAM)
		{
		}
	}
};

template <int INPUTS, int OUTPUTS>
__device__ void cuda_pulse(cuda_seayon& net, const typename dataset<INPUTS, OUTPUTS>::sample& sample)
{
	for (int n = 0; n < INPUTS; ++n)
		net.layers[0].neurons[n] = sample.inputs[n];

	for (int l2 = 1; l2 < net.layers.size(); ++l2)
	{
		const int l1 = l2 - 1;
		const int& n1count = net.layers[l1].nCount;
		const int& n2count = net.layers[l2].nCount;
		const ActivFunc_t func = net.layers[l2].activation;

		for (int n2 = 0; n2 < n2count; ++n2)
		{
			float z = 0;
			for (int n1 = 0; n1 < n1count; ++n1)
				z += net.layers[l2].weights[n2 * n1count + n1] * net.layers[l1].neurons[n1];
			z += net.layers[l2].biases[n2];

			net.layers[l2].neurons[n2] = func(z);
		}
	}
}

template <int INPUTS, int OUTPUTS>
__device__ void cuda_gradient_descent(const int& b, typename cuda_seayon::backprop_matrix<INPUTS, OUTPUTS>::device& mm)
{
	const int LASTL = mm.layers.size() - 1;
	typename cuda_seayon::backprop_matrix<INPUTS, OUTPUTS>::device::batch& thread = mm.batches[b];
	cuda_seayon& net = *thread.net;
	const float n = mm.n;
	const float m = mm.m;

	for (int i = 0; i < mm.batch_size; ++i)
	{
		const typename dataset<INPUTS, OUTPUTS>::sample& sample = *mm.sample_pointers[b * mm.batch_size + i];
		cuda_pulse<INPUTS, OUTPUTS>(net, sample);

		{
			const int& ncount = net.layers.back().nCount;
			const auto& deri = net.layers.back().derivative;

			for (int n2 = 0; n2 < ncount; ++n2)
			{
				thread.layers.back().deltas[n2] = deri(net.layers.back().neurons[n2]) * 2.0f * (net.layers.back().neurons[n2] - sample.outputs[n2]);
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
	}
}

template <int INPUTS, int OUTPUTS>
__global__ void kernel(typename cuda_seayon::backprop_matrix<INPUTS, OUTPUTS>::device* on_device)
{
	const int b = blockIdx.x * blockDim.x + threadIdx.x;

	cuda_gradient_descent<INPUTS, OUTPUTS>(b, *on_device);
}

template <int INPUTS, int OUTPUTS, int T_INPUTS, int T_OUTPUTS>
void cuda_seayon::mini_batch(const int& max_iterations, const float& n, const float& m, const int& batch_size, int total_threads, const dataset<INPUTS, OUTPUTS>& traindata, const dataset<T_INPUTS, T_OUTPUTS>& testdata)
{
	const int LASTL = layers.size() - 1;
	const int block_count = total_threads / 512 + 1;      // Optimal: power of 2
	const int thread_count = total_threads / block_count; // Optimal: multiple of 32, range 128 and 512
	total_threads = block_count * thread_count;
	const int samples.size() = batch_size * total_threads;

	// const int per_thread = traindata.samples.size() / batch_size / total_threads;
	// int batch_count = per_thread * total_threads;

	// total_threads = batch_count / per_thread;
	// int block_count = total_threads / 512 + 1;      // Optimal: power of 2
	// int thread_count = total_threads / block_count; // Optimal: multiple of 32, range 128 and 512
	// total_threads = block_count * thread_count;

	// int unused = traindata.samples.size() - total_threads * per_thread * batch_size;
	// total_threads += unused / (per_thread * batch_size);
	// block_count = total_threads / 512 + 1;
	// thread_count = total_threads / block_count;
	// total_threads = block_count * thread_count;

	// unused = traindata.samples.size() - total_threads * per_thread * batch_size;
	// batch_count = per_thread * total_threads;

	auto start = std::chrono::high_resolution_clock::now();

	size_t used_bytes;
	size_t used_by_bytes;
	size_t free_bytes;
	size_t before_free_bytes;
	size_t total_bytes;

	if (cudaMemGetInfo(&before_free_bytes, &total_bytes) != cudaSuccess)
	{
		printf("Failed to get gpu memory info");
		return;
	}

	backprop_matrix<INPUTS, OUTPUTS> mm(*this, traindata, total_threads, batch_size, n, m);

	if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess)
	{
		printf("Failed to get gpu memory info");
		return;
	}

	constexpr size_t mb = 1048576;
	used_bytes = (total_bytes - free_bytes) / mb;
	used_by_bytes = (before_free_bytes - free_bytes) / mb;
	free_bytes /= mb;

	std::chrono::milliseconds time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);

	printf("Cuda has launched in %lldms with:\n", time.count());
	printf("%i blocks | ", block_count);
	printf("%i threads | ", thread_count);
	printf("%i samples per thread | ", batch_size);
	printf("%i/%i unused samples\n\n", traindata.samples.size() - total_threads * batch_size, traindata.samples.size());

	printf("GPU memory usage: %llumb used | %llumb used by seayon | %llumb free\n", used_bytes, used_by_bytes, free_bytes);

	fitlog<T_INPUTS, T_OUTPUTS> logger(*this, traindata.samples.size(), testdata, max_iterations, printloss, logfolder);

	for (int run = 1; run <= max_iterations; ++run)
	{
		kernel<INPUTS, OUTPUTS> << <block_count, thread_count >> > (mm.on_device);

		cudaDeviceSynchronize();
		mm.sync();

		for (int b = 0; b < total_threads; ++b)
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

		logger.log(run);
	}

	printf("\n\n");
}