#include <iostream>
#include <omp.h>
#include <chrono>

int main()
{
    constexpr int size = 1000000;

    int *a = new int[size];
    int *b = new int[size];
    int *c = new int[size];

    for (int i = 0; i < size; ++i)
    {
        a[i] = 1;
        b[i] = 2;
    }

    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for
    for (int i = 0; i < size; ++i)
    {
        c[i] = a[i] + b[i];
        c[i] *= c[i];
        c[i] += 2;
    }

    std::chrono::microseconds elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);

    printf("1. %i - %lldus\n", c[0], (long long)elapsed.count());

    start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < size; ++i)
    {
        c[i] = a[i] + b[i];
        c[i] *= c[i];
        c[i] += 2;
    }

    elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);

    printf("2. %i - %lldus\n", c[0], (long long)elapsed.count());

    return 0;
}