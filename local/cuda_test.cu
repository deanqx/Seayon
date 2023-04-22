#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <chrono>

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

__host__ __device__
void add(const float* a, const float* b, float* c, const int& i)
{
    c[i] = a[i] + b[i];
}

__global__
void kernel(const float* a, const float* b, float* c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    add(a, b, c, i);
}

int main()
{
    constexpr int N = 10;

    int block_count = N / 512 + 1;         // Optimal: power of 2
    int thread_count = N / block_count;    // Optimal: multiple of 32, range 128 and 512
    int batch_count = block_count * thread_count;

    printf("block_count: %i\nthread_count: %i\nbatch_count: %i\n", block_count, thread_count, batch_count);

    float a[N];
    float b[N];
    float c[N];

    for (int i = 0; i < N; ++i)
    {
        a[i] = 7;
        b[i] = 3;
    }

    float* cudaA;
    float* cudaB;
    float* cudaC;

    cudaMalloc(&cudaA, N * sizeof(float));
    cudaMalloc(&cudaB, N * sizeof(float));
    cudaMalloc(&cudaC, N * sizeof(float));

    cudaMemcpy(cudaA, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaB, b, N * sizeof(float), cudaMemcpyHostToDevice);

    kernel << <block_count, thread_count >> > (cudaA, cudaB, cudaC);

    cudaDeviceSynchronize();

    cudaMemcpy(c, cudaC, N * sizeof(float), cudaMemcpyDeviceToHost);

    bool equal = true;
    for (int i = 0; equal && i < N; ++i)
    {
        if (c[i] != 10)
            equal = false;
    }
    printf("%f (%i)\n", c[0], (int)equal);

    return 0;
}