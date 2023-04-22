#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thread>
#include <iostream>
#include <chrono>

template <int N>
__host__ __device__ void work(const int* a, const int* b, int* c, int i)
{
    float j = 0.0f;
    for (int k = 0; k < 100000; ++k)
    {
        j += 0.1f;
    }

    c[i] = a[i] + b[i];
}

template <int N>
__global__ void kernel(const int* a, const int* b, int* c, const int per_thread)
{
    const int current = blockIdx.x * blockDim.x + threadIdx.x;
    const int begin = current * per_thread;
    const int end = begin + per_thread - 1;

    for (int i = begin; i <= end && i < N; ++i)
    {
        work<N>(a, b, c, i);
    }
}

void check(int* c, const int& N)
{
    bool equals = true;
    for (int i = 0; i < N && equals; ++i)
    {
        equals = (c[i] == 10);
    }

    printf("\tCorrect: %i (%i)\n", (int)equals, c[0]);
    memset(c, 0, N * sizeof(int));
}

template <int N>
void linear_test(const int* a, const int* b, int* c)
{
    auto linear_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; ++i)
        work<N>(a, b, c, i);

    std::chrono::microseconds linear = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - linear_start);
    printf("Linear:   \t%lli", linear.count());
    check(c, N);
    printf("\t->%lli\n", linear.count() / N);
}

template <int N, int THREADS, int PER_THREAD>
void parallel_test(const int* a, const int* b, int* c)
{
    auto para_start = std::chrono::high_resolution_clock::now();

    std::thread threads[THREADS];

    for (int t = 0; t < THREADS; ++t)
    {
        threads[t] = std::thread([&, t]
            {
                const int begin = t * PER_THREAD;
                const int end = begin + PER_THREAD - 1;

                for (int i = begin; i <= end && i < N; ++i)
                {
                    work<N>(a, b, c, i);
                }
            });
    }

    for (int t = 0; t < THREADS; ++t)
        threads[t].join();

    std::chrono::microseconds para = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - para_start);
    printf("Parallel:\t%lli", para.count());
    check(c, N);
    printf("\t->%lli\n", para.count() / N);
}

template <int N>
void cuda_test(const int* a, const int* b, int* c, const int per_thread)
{
    auto cuda_start = std::chrono::high_resolution_clock::now();

    const int block_count = N / per_thread / 512 + 1;      // Optimal: power of 2
    const int thread_count = N / per_thread / block_count; // Optimal: multiple of 32, range 128 and 512
    const int used = per_thread * block_count * thread_count;

    printf("\nblock_count: %i thread_count: %i per_thread: %i unused: %i\n\n", block_count, thread_count, per_thread, N - used);

    int* cudaA;
    int* cudaB;
    int* cudaC;

    cudaMalloc(&cudaA, used * sizeof(int));
    cudaMalloc(&cudaB, used * sizeof(int));
    cudaMalloc(&cudaC, used * sizeof(int));

    cudaMemcpy(cudaA, a, used * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaB, b, used * sizeof(int), cudaMemcpyHostToDevice);

    kernel<N> << <block_count, thread_count >> > (cudaA, cudaB, cudaC, 0);

    for (int i = used; i < N; ++i)
    {
        work<N>(a, b, c, i);
    }

    auto launched = std::chrono::high_resolution_clock::now();

    cudaDeviceSynchronize();

    cudaMemcpy(c, cudaC, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(cudaA);
    cudaFree(cudaB);
    cudaFree(cudaC);

    std::chrono::microseconds cuda = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - cuda_start);
    std::chrono::microseconds cuda_launched = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - launched);

    printf("Cuda:   \t%llius (launched: %lldus)", cuda.count(), cuda_launched.count());
    check(c, N);
    printf("\t->%llius\n", cuda.count() / N);
}

int main()
{
    constexpr int N = 12000;
    constexpr int THREADS = 30;
    constexpr int PER_THREAD = N / THREADS;

    int* a = new int[N];
    int* b = new int[N];
    int* c = new int[N];

    for (int i = 0; i < N; ++i)
    {
        a[i] = 7;
        b[i] = 3;
    }

    linear_test<N>(a, b, c);
    parallel_test<N, THREADS, PER_THREAD>(a, b, c);
    cuda_test<N>(a, b, c, 1);
    auto start = std::chrono::high_resolution_clock::now();
    cuda_test<N>(a, b, c, 2);
    std::chrono::microseconds stop = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
    printf("%llu\n", stop.count());
    // cuda_test<N>(a, b, c, 3);
    // cuda_test<N>(a, b, c, 4);
    // cuda_test<N>(a, b, c, 8);
    // cuda_test<N>(a, b, c, 12);

    return 0;
}