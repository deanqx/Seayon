#include <iostream>
#include <thread>
#include <chrono>

int main()
{
    std::thread threads[32];

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 32; ++i)
    {
        threads[i] = std::thread([=]
            {
                int x = i * i;
            });
    }

    printf("Launched %lldus\n", std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count());

    for (int i = 0; i < 32; ++i)
    {
        threads[i].join();
    }

    printf("End %lldus\n", std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count());
}