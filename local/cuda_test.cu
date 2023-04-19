#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

struct memory_manager
{
    float* host_bias_gradients;
    float* host_weight_gradients;

    float* bias_gradients;
    float* weight_gradients;

    memory_manager* device_this;

    const int batch_count;
    const int LAYERS;
    const int nCount;
    const int n2Count;

    size_t nSize;
    size_t wSize;

    memory_manager(const int host_batch_count, const int host_LAYERS, const int host_nCount, const int host_n2Count)
        : batch_count(host_batch_count), LAYERS(host_LAYERS), nCount(host_nCount), n2Count(host_n2Count)
    {
        nSize = host_batch_count * host_LAYERS * host_nCount * sizeof(float);
        wSize = host_batch_count * host_LAYERS * host_nCount * host_n2Count * sizeof(float);

        host_bias_gradients = (float*)malloc(nSize);
        host_weight_gradients = (float*)malloc(wSize);

        cudaMalloc(&bias_gradients, nSize);
        cudaMalloc(&weight_gradients, wSize);

        cudaMalloc(&device_this, sizeof(memory_manager));

        for (int b = 0; b < host_batch_count; ++b)
        {
            const int nIndex0 = b * host_LAYERS * host_nCount;
            const int wIndex0 = b * host_LAYERS * host_nCount * host_n2Count;

            for (int l = 0; l < host_LAYERS; ++l)
            {
                const int nIndex1 = nIndex0 + l * host_nCount;
                const int wIndex1 = wIndex0 + l * host_nCount * host_n2Count;

                for (int n1 = 0; n1 < host_nCount; ++n1)
                {
                    const int wIndex2 = wIndex1 + n1 * host_n2Count;

                    host_bias_gradients[nIndex1 + n1] = 7;

                    for (int n2 = 0; n2 < host_n2Count; ++n2)
                    {
                        host_weight_gradients[wIndex2 + n2] = 3;
                    }
                }
            }
        }

        cudaMemcpy(bias_gradients, host_bias_gradients, nSize, cudaMemcpyHostToDevice);
        cudaMemcpy(weight_gradients, host_weight_gradients, wSize, cudaMemcpyHostToDevice);
        cudaMemcpy(device_this, this, sizeof(memory_manager), cudaMemcpyHostToDevice);
    }

    void sync()
    {
        cudaMemcpy(host_bias_gradients, bias_gradients, nSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(host_weight_gradients, weight_gradients, wSize, cudaMemcpyDeviceToHost);
    }
};

__host__ __device__
void work(const int& b, memory_manager& mm)
{
    const int nIndex0 = b * mm.LAYERS * mm.nCount;
    const int wIndex0 = b * mm.LAYERS * mm.nCount * mm.n2Count;

    for (int l = 0; l < mm.LAYERS; ++l)
    {
        const int nIndex1 = nIndex0 + l * mm.nCount;
        const int wIndex1 = wIndex0 + l * mm.nCount * mm.n2Count;

        for (int n1 = 0; n1 < mm.nCount; ++n1)
        {
            const int wIndex2 = wIndex1 + n1 * mm.n2Count;

            for (int n2 = 0; n2 < mm.n2Count; ++n2)
            {
                // printf("[%i] [%i]\n", wIndex2 + n2, nIndex1 + n1);
                mm.weight_gradients[wIndex2 + n2] += mm.bias_gradients[nIndex1 + n1];
            }
        }
    }
}

__global__
void work_kernel(memory_manager* device_mm)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;

    work(b, *device_mm);
}

int main()
{
    constexpr int batch_size = 10;
    constexpr int LAYERS = 10;
    constexpr int nCount = 10;
    constexpr int n2Count = 10;

    int block_count = batch_size / 512 + 1;         // Optimal: power of 2
    int thread_count = batch_size / block_count;    // Optimal: multiple of 32, range 128 and 512
    int batch_count = block_count * thread_count;

    printf("block_count: %i\nthread_count: %i\nbatch_count: %i\n", block_count, thread_count, batch_count);

    memory_manager mm(batch_count, LAYERS, nCount, n2Count);

    work_kernel << <block_count, thread_count >> > (mm.device_this); // TODO test function pointer

    // for (int b = 0; b < batch_count; ++b)
    // {
    //     work(b, LAYERS, nCount, n2Count, mm.host_bias_gradients, mm.host_weight_gradients);
    // }

    cudaDeviceSynchronize();
    mm.sync();

    bool equal = true;
    for (int b = 0; equal && b < batch_count; ++b)
    {
        const int wIndex0 = b * LAYERS * nCount * n2Count;

        for (int l = 0; equal && l < LAYERS; ++l)
        {
            const int wIndex1 = wIndex0 + l * nCount * n2Count;

            for (int n1 = 0; equal && n1 < nCount; ++n1)
            {
                const int wIndex2 = wIndex1 + n1 * n2Count;

                for (int n2 = 0; equal && n2 < n2Count; ++n2)
                {
                    if (mm.host_weight_gradients[wIndex2 + n2] != 10)
                    {
                        printf("error: %i\n", wIndex2 + n2);
                        equal = false;
                    }
                }
            }
        }
    }
    printf("%f (%i)\n", mm.host_weight_gradients[0], (int)equal);

    return 0;
}