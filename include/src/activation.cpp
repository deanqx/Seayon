#include "../seayon.hpp"
#include <cmath>

float seayon::Sigmoid(const float z)
{
    return 1.0f / (1.0f + exp(-z));
}
float seayon::dSigmoid(const float a)
{
    return a * (1.0f - a);
}
float seayon::Tanh(const float z)
{
    const float x0 = exp(z);
    const float x1 = exp(-z);
    return (x0 - x1) / (x0 + x1);
}
float seayon::dTanh(const float a)
{
    const float t = seayon::Tanh(a);
    return 1 - t * t;
}
float seayon::ReLu(const float z)
{
    return (z < 0.0f ? 0.0f : z);
}
float seayon::dReLu(const float a)
{
    return (a < 0.0f ? 0.0f : 1.0f);
}
float seayon::LeakyReLu(const float z)
{
    float x = 0.1f * z;
    return (x > z ? x : z);
}
float seayon::dLeakyReLu(const float a)
{
    return (a > 0.0f ? 1.0f : 0.01f);
}

float* seayon::model::pulse(const float* inputs)
{
    memcpy(layers[0].neurons, inputs, xsize * sizeof(float));

    for (int l2 = 1; l2 < layerCount; ++l2)
    {
        const int l1 = l2 - 1;
        const int& n1count = layers[l1].nCount;
        const int& n2count = layers[l2].nCount;
        const auto& func = layers[l2].activation;

        for (int n2 = 0; n2 < n2count; ++n2)
        {
            const int row = n2 * n1count;

            float z = 0;
            for (int n1 = 0; n1 < n1count; ++n1)
                z += layers[l2].weights[row + n1] * layers[l1].neurons[n1];
            z += layers[l2].biases[n2];

            layers[l2].neurons[n2] = func(z);
        }
    }

    return layers[layerCount - 1].neurons;
}