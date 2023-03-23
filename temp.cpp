#include <math.h>
#include <vector>

inline float Sigmoid(const float& z)
{
    return 1.0f / (1.0f + exp(-z));
}

inline float randf(float min, float max)
{
    return min + (float)rand() / (float)(RAND_MAX / (max - min));
}

struct layer
{
    int nCount;
    int wCount;

    std::vector<float> weights;
    std::vector<float> neurons;
    std::vector<float> biases;

    layer(const int PREVIOUS, const int LAYERS)
    {
        nCount = LAYERS;
        wCount = LAYERS * PREVIOUS;
        weights.resize(wCount);
        neurons.resize(LAYERS);
        biases.resize(LAYERS);

        for (int i = 0; i < wCount; ++i)
        {
            weights[i] = randf(-2.0f, 2.0f);
        }
    }
};

int main()
{
    constexpr int LAYERS = 4;
    constexpr int INPUTS = 784;
    float inputs[INPUTS]{};

    layer layers[LAYERS]{ {0, 784}, {784, 16}, {16, 16}, {16, 10} };

    for (int n = 0; n < INPUTS; ++n)
        layers[0].neurons[n] = inputs[n];

    const int layerCount = LAYERS - 1;
    for (int l1 = 0; l1 < layerCount; ++l1)
    {
        const int l2 = l1 + 1;
        const int& ncount = layers[l2].nCount;

        for (int n2 = 0; n2 < ncount; ++n2)
        {
            float z = 0;
            for (int n1 = 0; n1 < layers[l1].nCount; ++n1)
                z += layers[l2].weights[n2 + n1 * ncount] * layers[l1].neurons[n1];
            z += layers[l2].biases[n2];

            layers[l2].neurons[n2] = Sigmoid(z);
        }
    }

    return 0;
}