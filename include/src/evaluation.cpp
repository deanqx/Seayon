#include "../seayon.hpp"
#include <algorithm>

float seayon::model::loss(const typename dataset::sample& sample)
{
    pulse(sample.x);

    const int LASTL = layerCount - 1;

    float d = 0.0;
    for (int i = 0; i < ysize; ++i)
    {
        const float x = layers[LASTL].neurons[i] - sample.y[i];
        d += x * x;
    }

    return d / (float)ysize;
}
float seayon::model::loss(const dataset& data)
{
    if (!check(data))
    {
        printf("\tCurrupt training data!\n");
        return .0f;
    }

    float d = 0.0;
    for (int i = 0; i < data.size(); ++i)
    {
        d += loss(data[i]);
    }

    return d / (float)data.size();
}
float seayon::model::diff(const typename dataset::sample& sample, std::vector<float> factor)
{
    pulse(sample.x);

    const int LASTL = layerCount - 1;

    float d = 0.0;
    for (int i = 0; i < ysize; ++i)
    {
        d += std::abs((layers[LASTL].neurons[i] * factor[i]) - sample.y[i]);
    }

    return d / (float)ysize;
}
float seayon::model::diff(const dataset& data, std::vector<std::vector<float>> factors)
{
    if (!check(data))
    {
        printf("\tCurrupt training data!\n");
        return .0f;
    }

    if (factors.size() == 0)
        factors = std::vector<std::vector<float>>(data.size(), std::vector<float>(ysize, 1.0f));

    float d = 0.0f;
    for (int i = 0; i < data.size(); ++i)
    {
        d += diff(data[i], factors[i]);
    }

    return d / (float)data.size();
}
float seayon::model::diff_max(const typename dataset::sample& sample, std::vector<float> factor)
{
    pulse(sample.x);

    const int LASTL = layerCount - 1;

    float d = std::abs((layers[LASTL].neurons[0] * factor[0]) - sample.y[0]);
    for (int i = 1; i < ysize; ++i)
    {
        const float x = std::abs((layers[LASTL].neurons[i] * factor[i]) - sample.y[i]);
        if (d < x)
            d = x;
    }

    return d;
}
float seayon::model::diff_max(const dataset& data, std::vector<std::vector<float>> factors)
{
    if (!check(data))
    {
        printf("\tCurrupt training data!\n");
        return .0f;
    }

    if (factors.size() == 0)
        factors = std::vector<std::vector<float>>(data.size(), std::vector<float>(ysize, 1.0f));

    float d = diff_max(data[0], factors[0]);
    for (int i = 0; i < data.size(); ++i)
    {
        const float x = diff_max(data[i], factors[i]);
        if (d < x)
            d = x;
    }

    return d;
}
float seayon::model::diff_min(const typename dataset::sample& sample, std::vector<float> factor)
{
    pulse(sample.x);

    const int LASTL = layerCount - 1;

    float d = std::abs((layers[LASTL].neurons[0] * factor[0]) - sample.y[0]);
    for (int i = 1; i < ysize; ++i)
    {
        const float x = std::abs((layers[LASTL].neurons[i] * factor[i]) - sample.y[i]);
        if (d < x)
            d = x;
    }

    return d;
}
float seayon::model::diff_min(const dataset& data, std::vector<std::vector<float>> factors)
{
    if (!check(data))
    {
        printf("\tCurrupt training data!\n");
        return .0f;
    }

    if (factors.size() == 0)
        factors = std::vector<std::vector<float>>(data.size(), std::vector<float>(ysize, 1.0f));

    float d = diff_min(data[0], factors[0]);
    for (int i = 1; i < data.size(); ++i)
    {
        const float x = diff_min(data[i], factors[i]);
        if (d > x)
            d = x;
    }

    return d;
}
float seayon::model::accruacy(const dataset& data)
{
    const int LASTL = layerCount - 1;

    if (!check(data))
    {
        printf("\tCurrupt training data!\n");
        return .0f;
    }

    float a = 0;
    for (int i = 0; i < data.size(); ++i)
    {
        pulse(data[i].x);

        if (std::max_element(layers[LASTL].neurons, layers[LASTL].neurons + layers[LASTL].nCount) - layers[LASTL].neurons
            == std::max_element(data[i].y, data[i].y + ysize) - data[i].y)
        {
            ++a;
        }
    }

    return a / (float)data.size();
}