#include <thread>
#include <iostream>
#include <chrono>

template <int N>
void work(const int id, const int per_thread, const int* a, const int* b, int* c)
{
    const int begin = id * per_thread;
    const int end = begin + per_thread - 1;

    for (int i = begin; i <= end && i < N; ++i)
    {
        float j = 0.0f;
        for (int k = 0; k < 1000000; ++k)
        {
            j += 0.1f;
        }

        c[i] = a[i] + b[i];
    }
}

void check(const int* c, const int& N)
{
    bool equals = true;
    for (int i = 0; i < N && equals; ++i)
    {
        equals = (c[i] == 10);
    }

    printf("\tCorrect: %i -> %i\n", (int)equals, c[0]);
}

template <int N>
void linear_test(const int* a, const int* b, int* c)
{
    auto linear_start = std::chrono::high_resolution_clock::now();

    work<N>(0, N, a, b, c);

    std::chrono::microseconds linear = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - linear_start);
    printf("Linear: %lli", linear.count());
    check(c, N);
}

template <int N, int THREADS, int PER_THREAD>
void paralell_test(const int* a, const int* b, int* c)
{
    auto para_start = std::chrono::high_resolution_clock::now();

    std::thread threads[THREADS];

    for (int i = 0; i < THREADS; ++i)
    {
        threads[i] = std::thread(work<N>, i, PER_THREAD, a, b, c);
    }

    for (int i = 0; i < THREADS; ++i)
        threads[i].join();

    std::chrono::microseconds para = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - para_start);
    printf("Paralell: %lli", para.count());
    check(c, N);
}

template <int N, int CUDA_THREADS, int CUDA_PER_THREAD>
void cuda_test(const int* a, const int* b, int* c)
{
    auto cuda_start = std::chrono::high_resolution_clock::now();

    // TODO cuda test

    std::chrono::microseconds cuda = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - cuda_start);
    printf("Cuda: %lli", cuda.count());
    check(c, N);
}

int main()
{
    constexpr int N = 1200;
    constexpr int THREADS = 30;
    constexpr int PER_THREAD = N / THREADS;
    constexpr int CUDA_THREADS = 30;
    constexpr int CUDA_PER_THREAD = N / THREADS;

    int a[N];
    int b[N];
    int c[N];

    for (int i = 0; i < N; ++i)
    {
        a[i] = 7;
        b[i] = 3;
    }

    linear_test<N>(a, b, c);
    paralell_test<N, THREADS, PER_THREAD>(a, b, c);
    cuda_test<N, CUDA_THREADS, CUDA_PER_THREAD>(a, b, c);

    return 0;
}