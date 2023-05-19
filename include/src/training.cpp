#include "../seayon.hpp"
#include <random>
#include <thread>
#include "backprop_matrix.cpp"
#include "fitlog.cpp"

void seayon::model::fit(const dataset& traindata, const dataset& testdata, const bool shuffle,
    int max_epochs, int batch_size, int thread_count, float learning_rate, float beta1, float beta2, float epsilon)
{ // WARN thread_count always improves performance(probably loosing data on the way)
    if (!check(traindata) || !check(testdata))
    {
        printf("\tCurrupt training data!\n");
        return;
    }

    if (batch_size < 1)
        batch_size = 1;

    if (thread_count < 1)
        thread_count = 1;

    const int batch_count = traindata.samples.size() / batch_size;
    const int sampleCount = batch_size * batch_count;

    backprop_matrix matrix(thread_count, *this, traindata);

    fitlog logger(*this, sampleCount, testdata, max_epochs, printloss, logfolder);

    if (thread_count == 1)
    {
        matrix.threads[0].net.reset(this, [](seayon::model* obj) {});

        for (int epoch = 1; epoch <= max_epochs; ++epoch)
        {
            for (int b = 0; b < batch_count; ++b)
            {
                const int row = b * batch_size;

                for (int i = 0; i < batch_size; ++i)
                {
                    matrix.threads[0].backprop(traindata[row + i]);
                }

                matrix.threads[0].apply(learning_rate, beta1, beta2, epsilon, (float)batch_size);
            }

            if (shuffle)
                matrix.shuffle();

            if (logger.log(epoch) == 0.0f)
                break;
        }
    }
    else
    {
        const int per_thread = batch_count / thread_count;
        const int unused_begin = thread_count * per_thread * batch_size;
        const int unused_end = traindata.samples.size() - unused_begin - 1;

        std::vector<std::thread> threads(thread_count);

        for (int epoch = 1; epoch <= max_epochs; ++epoch)
        {
            for (int t = 0; t < thread_count; ++t)
            {
                threads[t] = std::thread([&, t]
                    {
                        const int begin = t * per_thread;
                        const int end = begin + per_thread - 1;

                        for (int b = begin; b <= end; ++b)
                        {
                            const int row = b * batch_size;

                            for (int i = 0; i < batch_size; ++i)
                            {
                                matrix.threads[t].backprop(traindata[row + i]);
                            }

                            matrix.threads[t].apply(learning_rate, beta1, beta2, epsilon, (float)batch_size);
                        }
                    });
            }

            for (int t = 0; t < thread_count; ++t)
            {
                threads[t].join();
            }

            matrix.sync(*this);
            if (shuffle)
                matrix.shuffle();

            if (logger.log(epoch) == 0.0f)
                break;
        }
    }

    printf("\n\n");
}