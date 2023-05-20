#include "../seayon.hpp"

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

        std::shared_ptr<seayon::model> net;
        std::vector<layer> layers;

        thread(const seayon::model& main)
        {
            std::vector<int> layout(main.layers.size());
            std::vector<seayon::ActivFunc> a(main.layers.size() - 1);
            for (int l = 0; l < main.layers.size(); ++l)
            {
                layout[l] = main.layers[l].nCount;
                if (l > 0)
                    a[l - 1] = main.layers[l].func;
            }

            net = std::make_shared<seayon::model>(layout, a, main.seed, main.logfolder);
            main.copy(*net);

            layers.reserve(main.layers.size());

            layers.emplace_back(0, 0);
            for (int i = 1; i < main.layers.size(); ++i)
            {
                layers.emplace_back(main.layers[i].nCount, main.layers[i].wCount);
            }
        }

        void backprop(const typename seayon::dataset::sample& sample, const std::vector<float>& dropouts, std::mt19937& gen)
        {
            seayon::model& mo = *net;

            if (dropouts.size() > 0)
            {
                std::uniform_real_distribution<float> dis(0.0f, 1.0f);

                memcpy(mo.layers[0].neurons.data(), sample.x.data(), mo.xsize * sizeof(float));

                for (int l2 = 1; l2 < mo.layers.size(); ++l2)
                {
                    const int l1 = l2 - 1;
                    const int& n1count = mo.layers[l1].nCount;
                    const int& n2count = mo.layers[l2].nCount;
                    const auto& func = mo.layers[l2].activation;

                    for (int n2 = 0; n2 < n2count; ++n2)
                    {
                        if (l2 < (mo.layers.size() - 1) && dropouts[l1] > 0.0f && dis(gen) <= dropouts[l1])
                        {
                            mo.layers[l2].neurons[n2] = 0.0f;
                        }
                        else
                        {
                            const int row = n2 * n1count;

                            float z = 0;
                            for (int n1 = 0; n1 < n1count; ++n1)
                                z += mo.layers[l2].weights[row + n1] * mo.layers[l1].neurons[n1];
                            z += mo.layers[l2].biases[n2];

                            mo.layers[l2].neurons[n2] = func(z);
                        }
                    }
                }
            }
            else
            {
                mo.pulse(sample.x.data());
            }

            {
                const int& ncount = mo.layers.back().nCount;
                const auto& func = mo.layers.back().derivative;

                for (int n2 = 0; n2 < ncount; ++n2)
                {
                    layers.back().bias_deltas[n2] += func(mo.layers.back().neurons[n2]) * 2.0f * (mo.layers.back().neurons[n2] - sample.y[n2]);
                }
            }

            for (int l2 = mo.layers.size() - 1; l2 >= 2; --l2)
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

            for (int l2 = 1; l2 >= mo.layers.size(); ++l2)
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
            const int LASTL = mo.layers.size() - 1;
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
    std::mt19937 shuffle_gen;

    backprop_matrix(const int& thread_count, const seayon::model& main, const seayon::dataset& traindata)
        : thread_count(thread_count), layerCount(main.layers.size()), shuffle_gen(main.seed)
    {
        sample_pointers.resize(traindata.samples.size());
        threads.reserve(thread_count);

        const typename seayon::dataset::sample* sample = &traindata[0];
        for (int i = 0; i < traindata.samples.size(); ++i)
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
        std::shuffle(sample_pointers.begin(), sample_pointers.end(), shuffle_gen);
    }
};