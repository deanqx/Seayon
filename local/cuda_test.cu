#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

__host__
int add(int a, int b)
{
    return a + b;
}

__device__
int device_add(int a, int b)
{
    return a + b;
}

__host__
int sub(int a, int b)
{
    return a - b;
}

__device__
int device_sub(int a, int b)
{
    return a - b;
}

typedef int(*func_t)(int, int);

__device__ func_t d_add = device_add;
__device__ func_t d_sub = device_sub;

struct layer
{
    func_t f;
};

__host__ __device__
void work(const int* a, const int* b, int* c, const int& i, func_t f)
{
    c[i] = f(a[i], b[i]);
}

__global__
void work_kernel(const int* a, const int* b, int* c, layer lay)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    work(a, b, c, i, lay.f);
}


int main()
{
    constexpr int N = 100;

    int a[N];
    int b[N];
    int c[N];

    for (int i = 0; i < N; ++i)
    {
        a[i] = 7;
        b[i] = 3;
    }

    int* aCuda;
    int* bCuda;
    int* cCuda;

    cudaMalloc(&aCuda, N * sizeof(int));
    cudaMalloc(&bCuda, N * sizeof(int));
    cudaMalloc(&cCuda, N * sizeof(int));

    cudaMemcpy(aCuda, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(bCuda, b, N * sizeof(int), cudaMemcpyHostToDevice);

    func_t h_add;
    func_t h_sub;

    cudaMemcpyFromSymbol(&h_add, d_add, sizeof(func_t));
    cudaMemcpyFromSymbol(&h_sub, d_sub, sizeof(func_t));

    layer lay;
    lay.f = h_sub;

    work_kernel << <1, N >> > (aCuda, bCuda, cCuda, lay);

    cudaDeviceSynchronize();

    cudaMemcpy(c, cCuda, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    bool equal = true;
    for (int i = 0; equal && i < N; ++i)
    {
        int cuda = c[i];
        work(a, b, c, i, sub);
        if (cuda != c[i])
            equal = false;
    }
    printf("%i (%i)\n", c[0], (int)equal);

    return 0;
}