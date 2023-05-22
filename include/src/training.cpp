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
    float steps_per_epoch,
    float learning_rate,
    std::vector<float> dropouts,
    step_callback_t callback,
    float beta1,
    float beta2,
    float epsilon)
{ // WARN thread_count always improves performance(probably loosing data on the way)
    if (!check(traindata) || !check(testdata))
    {
        printf("\t--- error: Currupt training data! ---\n");
        return;
    }

    if (dropouts.size() != 0)
    {
        if (dropouts.size() > layers.size() - 2)
        {
            printf("\t--- error: can't dropout first or last layer (%llu == %llu) ---\n", dropouts.size(), layers.size() - 2);
            return;
        }

        if (dropouts.size() < layers.size() - 2)
        {
            dropouts.resize(layers.size() - 2);
        }
    }


    if (batch_size < 1)
        batch_size = 1;

    const bool val_loss = (traindata.samples.size() != testdata.samples.size());
    const int batch_count = (float)(traindata.samples.size() / batch_size) * steps_per_epoch;
    const int sampleCount = batch_size * batch_count;

    if (verbose == 2 || verbose == 4)
    {
        printf("--> Training with:\n");
        printf("traindata          %i/%llu samples\n", sampleCount, traindata.samples.size());
        if (val_loss)
            printf("testdata           %llu samples\n", testdata.samples.size());
        else
            printf("testdata           Disabled\n");
        printf("epochs             %i\n", epochs);
        printf("threads            %i\n", batch_size);
        printf("iterations         %i <- %i\n", batch_count, sampleCount);
        printf("shuffle            %s\n", shuffle ? "True" : "False");
        printf("steps per epoch    %.1f%%\n", steps_per_epoch * 100.0f);
        printf("learning rate      %f\n", learning_rate);
        printf("dropout            %s\n", dropouts.size() > 0 ? "Enabled" : "Disabled");
        printf("beta1              %f\n", beta1);
        printf("beta2              %f\n", beta2);
        printf("epsilon            %f\n", epsilon);
    }

    backprop_matrix matrix(*this, traindata, batch_size);

    fitlog logger(*this, sampleCount, traindata, testdata, epochs, val_loss, verbose, logfolder, callback);

    if (batch_size == 1)
    {
        std::mt19937 gen(seed);

        for (int epoch = 1; epoch <= epochs; ++epoch)
        {
            if (shuffle)
                matrix.shuffle();

            float loss_sum = 0.0f;

            for (int i = 0; i < sampleCount; ++i)
            {
                loss_sum += matrix.threads[0].backprop_async(traindata[i], dropouts, gen);
                matrix.threads[0].apply(learning_rate, beta1, beta2, epsilon);
            }

            if (logger.log(epoch, loss_sum / sampleCount))
                break;
        }
    }
    else
    {
        std::vector<std::thread> threads(batch_size);
        std::vector<float> losses(batch_size);
        std::vector<std::mt19937> gens;
        gens.reserve(batch_size);

        for (int i = 0; i < batch_size; ++i)
        {
            gens.emplace_back(seed);
        }

        for (int epoch = 1; epoch <= epochs; ++epoch)
        {
            if (shuffle)
                matrix.shuffle();

            memset(losses.data(), 0, losses.size() * sizeof(float));

            for (int b = 0; b < batch_count; ++b)
            {
                const int offset = b * batch_size;

                for (int i = 0; i < batch_size; ++i)
                {
                    threads[i] = std::thread([&, i]
                        {
                            losses[i] += matrix.threads[i].backprop_async(traindata[offset + i], dropouts, gens[i]);
                        });
                }

                for (int i = 0; i < batch_size; ++i)
                {
                    threads[i].join();
                    matrix.threads[i].apply(learning_rate, beta1, beta2, epsilon);
                }

                matrix.sync();
            }

            if (logger.log(epoch, std::accumulate(losses.begin(), losses.end(), 0.0f) / sampleCount))
                break;
        }
    }

    printf("\n\n");
}