#include "../seayon.hpp"
#include <thread>
#include "backprop_matrix.cpp"
#include "fitlog.cpp"

void seayon::model::fit(const dataset& traindata,
    const dataset& testdata,
    int epochs,
    int batch_size,
    int verbose,
    bool shuffle,
    int steps_per_epoch,
    int thread_count,
    float learning_rate,
    std::vector<float> dropouts,
    float beta1,
    float beta2,
    float epsilon)
{ // WARN thread_count always improves performance(probably loosing data on the way)
    if (!check(traindata) || !check(testdata))
    {
        printf("\t--- error: Currupt training data! ---\n");
        return;
    }

    if (dropouts.size() != 0 && dropouts.size() != layers.size() - 2)
    {
        printf("\t--- error: can't dropout first or last layer (%llu == %llu) ---\n", dropouts.size(), layers.size() - 2);
        return;
    }

    if (batch_size < 1)
        batch_size = 1;

    if (thread_count < 1)
        thread_count = 1;

    const int batch_count = traindata.samples.size() / batch_size;
    const int sampleCount = batch_size * batch_count;

    if (steps_per_epoch < 1 || steps_per_epoch > batch_count)
        steps_per_epoch = batch_count;

    if (verbose > 1)
    {
        printf("--> Training with:\n");
        printf("traindata          %llu samples\n", traindata.samples.size());
        if (traindata.samples.size() == testdata.samples.size())
            printf("testdata           Disabled\n");
        else
            printf("testdata           %llu samples\n", testdata.samples.size());
        printf("shuffle            %s\n", shuffle ? "True" : "False");
        printf("steps_per_epoch    %i\n", steps_per_epoch);
        printf("epochs             %i\n", epochs);
        printf("batch_size         %i\n", batch_size);
        printf("thread_count       %i\n", thread_count);
        printf("learning_rate      %f\n", learning_rate);
        printf("beta1              %f\n", beta1);
        printf("beta2              %f\n", beta2);
        printf("epsilon            %f\n", epsilon);
    }

    backprop_matrix matrix(thread_count, *this, traindata);

    fitlog logger(*this, sampleCount, traindata, testdata, epochs, (traindata.samples.size() != testdata.samples.size()), logfolder);

    if (thread_count == 1)
    {
        matrix.threads[0].net.reset(this, [](seayon::model* obj) {});

        std::mt19937 gen(seed);

        for (int epoch = 1; epoch <= epochs; ++epoch)
        {
            if (shuffle)
                matrix.shuffle();

            for (int b = 0; b < steps_per_epoch; ++b)
            {
                const int row = b * batch_size;

                for (int i = 0; i < batch_size; ++i)
                {
                    matrix.threads[0].backprop(traindata[row + i], dropouts, gen);
                }

                matrix.threads[0].apply(learning_rate, beta1, beta2, epsilon, (float)batch_size);
            }

            if (verbose > 0)
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
        std::vector<std::mt19937> gens;

        for (int i = 0; i < thread_count; ++i)
        {
            gens.emplace_back(seed);
        }

        for (int epoch = 1; epoch <= epochs; ++epoch)
        {
            if (shuffle)
                matrix.shuffle();

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
                                matrix.threads[t].backprop(traindata[row + i], dropouts, gens[t]);
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

            if (verbose > 0)
                if (logger.log(epoch) == 0.0f)
                    break;
        }
    }

    printf("\n\n");
}