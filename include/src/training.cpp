#include "../seayon.hpp"
#include <random>
#include <thread>

struct backprop_matrix
{
    struct thread
    {
        struct layer
        {
            std::vector<float> bias_deltas;
            std::vector<float> bias_velocities;
            std::vector<float> bias_rms;
            std::vector<float> weight_deltas;
            std::vector<float> weight_velocities;
            std::vector<float> weight_rms;

            const int nCount;
            const int wCount;

            layer(const int& nCount, const int& wCount)
                : nCount(nCount), wCount(wCount)
            {
                bias_deltas.resize(nCount);
                bias_velocities.resize(nCount);
                bias_rms.resize(nCount);
                weight_deltas.resize(wCount);
                weight_velocities.resize(wCount);
                weight_rms.resize(wCount);
            }
        };

        std::unique_ptr<seayon::model> net;
        std::vector<layer> layers;

        thread(const seayon::model& main)
        {
            std::vector<int> layout(main.layerCount);
            std::vector<seayon::ActivFunc> a(main.layerCount - 1);
            for (int l = 0; l < main.layerCount; ++l)
            {
                layout[l] = main.layers[l].nCount;
                if (l > 0)
                    a[l - 1] = main.layers[l].func;
            }

            net = std::make_unique<seayon::model>(layout, a, main.seed, main.printloss, main.logfolder);
            main.copy(*net);

            layers.reserve(main.layerCount);

            layers.emplace_back(0, 0);
            for (int i = 1; i < main.layerCount; ++i)
            {
                layers.emplace_back(main.layers[i].nCount, main.layers[i].wCount);
            }
        }

        void backprop(const typename seayon::dataset::sample& sample)
        {
            seayon::model& mo = *net;
            const int LASTL = mo.layerCount - 1;

            mo.pulse(sample.x);

            {
                const int& ncount = mo.layers[LASTL].nCount;
                const auto& func = mo.layers[LASTL].derivative;

                for (int n2 = 0; n2 < ncount; ++n2)
                {
                    layers[LASTL].bias_deltas[n2] += func(mo.layers[LASTL].neurons[n2]) * 2.0f * (mo.layers[LASTL].neurons[n2] - sample.y[n2]);
                }
            }

            for (int l2 = LASTL; l2 >= 2; --l2)
            {
                const int l1 = l2 - 1;
                const int& n1count = mo.layers[l1].nCount;
                const int& n2count = mo.layers[l2].nCount;
                const auto& func = mo.layers[l1].derivative;

                for (int n1 = 0; n1 < n1count; ++n1)
                {
                    float delta = 0;
                    for (int n2 = 0; n2 < n2count; ++n2)
                        delta += mo.layers[l2].weights[n2 * n1count + n1] * layers[l2].bias_deltas[n2];

                    layers[l1].bias_deltas[n1] += func(mo.layers[l1].neurons[n1]) * delta;
                }
            }

            for (int l2 = LASTL; l2 >= 1; --l2)
            {
                const int l1 = l2 - 1;
                const int& n1count = mo.layers[l1].nCount;
                const int& n2count = mo.layers[l2].nCount;

                for (int n2 = 0; n2 < n2count; ++n2)
                {
                    const float& db = layers[l2].bias_deltas[n2];

                    const int row = n2 * n1count;
                    for (int n1 = 0; n1 < n1count; ++n1)
                    {
                        const int windex = row + n1;
                        layers[l2].weight_deltas[windex] += db * mo.layers[l1].neurons[n1];
                    }
                }
            }
        }

        void apply(const float& alpha, const float& beta1, const float& beta2, const float& epsilon, const float batch_size)
        {
            seayon::model& mo = *net;
            const int LASTL = mo.layerCount - 1;
            const float ibeta1 = 1.0f - beta1;
            const float ibeta2 = 1.0f - beta2;

            for (int l2 = LASTL; l2 >= 1; --l2)
            {
                const int l1 = l2 - 1;
                const int& n1count = mo.layers[l1].nCount;
                const int& n2count = mo.layers[l2].nCount;

                for (int n2 = 0; n2 < n2count; ++n2)
                {
                    const float& db = layers[l2].bias_deltas[n2] / batch_size;

                    layers[l2].bias_velocities[n2] = beta1 * layers[l2].bias_velocities[n2] + ibeta1 * db;
                    layers[l2].bias_rms[n2] = beta2 * layers[l2].bias_rms[n2] + ibeta2 * db * db;
                    mo.layers[l2].biases[n2] -= alpha * (layers[l2].bias_velocities[n2] / (sqrt(layers[l2].bias_rms[n2]) + epsilon));

                    const int row = n2 * n1count;
                    for (int n1 = 0; n1 < n1count; ++n1)
                    {
                        const int windex = row + n1;
                        const float& dw = layers[l2].weight_deltas[windex] / batch_size;

                        layers[l2].weight_velocities[windex] = beta1 * layers[l2].weight_velocities[windex] + ibeta1 * dw;
                        layers[l2].weight_rms[windex] = beta2 * layers[l2].weight_rms[windex] + ibeta2 * dw * dw;
                        mo.layers[l2].weights[windex] -= alpha * (layers[l2].weight_velocities[windex] / (sqrt(layers[l2].weight_rms[windex]) + epsilon));
                    }
                }

                memset(layers[l2].bias_deltas.data(), 0, layers[l2].nCount * sizeof(float));
                memset(layers[l2].weight_deltas.data(), 0, layers[l2].wCount * sizeof(float));
            }
        }
    };

    std::vector<const typename seayon::dataset::sample*> sample_pointers;
    std::vector<thread> threads;
    const int thread_count;
    const int layerCount;

    backprop_matrix(const int& sampleCount, const int& thread_count, const seayon::model& main, const seayon::dataset& traindata)
        : thread_count(thread_count), layerCount(main.layerCount)
    {
        sample_pointers.resize(sampleCount);
        threads.reserve(thread_count);

        const typename seayon::dataset::sample* sample = &traindata[0];
        for (int i = 0; i < sampleCount; ++i)
        {
            sample_pointers[i] = sample + i;
        }

        for (int i = 0; i < thread_count; ++i)
        {
            threads.emplace_back(main);
        }
    }

    void sync(seayon::model& main)
    {
        std::vector<seayon::model*> links(thread_count);
        for (int i = 0; i < thread_count; ++i)
        {
            links[i] = threads[i].net.get();
        }

        main.combine_into(links.data(), thread_count);

        for (int i = 0; i < thread_count; ++i)
        {
            main.copy(*links[i]);
        }
    }

    void shuffle()
    {
        std::random_device rm_seed;
        std::shuffle(sample_pointers.begin(), sample_pointers.end(), std::mt19937(rm_seed()));
    }
};

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

    const int batch_count = traindata.size() / batch_size;
    const int per_thread = batch_count / thread_count;
    const int sampleCount = batch_size * batch_count;

    const int unused_begin = thread_count * per_thread * batch_size;
    const int unused_end = traindata.size() - unused_begin - 1;

    std::vector<std::thread> threads(thread_count);

    backprop_matrix matrix(sampleCount, thread_count, *this, traindata);

    fitlog logger(*this, sampleCount, testdata, max_epochs, printloss, logfolder);

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
                        for (int i = 0; i < batch_size; ++i)
                        {
                            matrix.threads[t].backprop(traindata[b * batch_size + i]);
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

    printf("\n\n");
}