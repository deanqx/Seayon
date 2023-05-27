#include "../seayon.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

void seayon::model::summary()
{
    int paras = 0;
    for (int i = 1; i < layers.size(); ++i)
    {
        paras += layers[i].wCount;
        paras += layers[i].nCount;
    }

    printf("--> Generating model with:\n");
    printf("parameters  %i\n", paras);
    printf("seed        %i\n", this->seed);
}

float seayon::model::evaluate(const dataset& data)
{
    printf("Loss           %f\n", loss(data));
    const float d = diff(data);
    printf("Difference     %f\n", d);
    printf("Max Difference %f\n", diff_max(data));
    printf("Min Difference %f\n", diff_min(data));
    printf("Accruacy       %.1f%%\n", accruacy(data) * 100.0f);
    printf("-----------------------------------------------\n\n");

    return d;
}

void seayon::model::print()
{
    HANDLE cmd = GetStdHandle(STD_OUTPUT_HANDLE);

    int normalColor;

    for (int l1 = 0; l1 < layers.size(); ++l1)
    {
        const int l2 = l1 + 1;

        if (l1 == 0)
        {
            normalColor = 7;
            SetConsoleTextAttribute(cmd, 7);

            printf("\nInput Layer:\n");
        }
        else if (l1 == layers.size() - 1)
        {
            normalColor = 11;
            SetConsoleTextAttribute(cmd, 11);

            printf("Output Layer:\n");
        }
        else
        {
            normalColor = 8;
            SetConsoleTextAttribute(cmd, 8);

            printf("Hidden Layer[%i]:\n", l1 - 1);
        }

        size_t largest = std::max_element(layers[l1].neurons.begin(), layers[l1].neurons.end()) - layers[l1].neurons.begin();
        for (int n1 = 0; n1 < layers[l1].nCount; ++n1)
        {
            printf("       Neuron[%02i]   ", n1);
            if (l1 == layers.size() - 1)
            {
                if (n1 == largest)
                    SetConsoleTextAttribute(cmd, 95);
                else
                    SetConsoleTextAttribute(cmd, 7);
                printf("%.2f", layers[l1].neurons[n1]);
                SetConsoleTextAttribute(cmd, normalColor);
            }
            else
                printf("%0.2f", layers[l1].neurons[n1]);

            if (l1 > 0)
            {
                if (layers[l1].biases[n1] <= 0.0f)
                {
                    printf("\t(");
                    SetConsoleTextAttribute(cmd, 12);
                    printf("%0.2f", layers[l1].biases[n1]);
                    SetConsoleTextAttribute(cmd, normalColor);
                    printf(")\n");
                }
                else
                    printf("\t(%0.2f)\n", layers[l1].biases[n1]);
            }
            else
                printf("\n");

            if (l2 < layers.size())
                for (int n2 = 0; n2 < layers[l2].nCount; ++n2)
                {
                    printf("         Weight[%02i] ", n2);
                    float& w = layers[l2].weights[n2 * layers[l1].nCount + n1];

                    if (w <= 0.0f)
                    {
                        SetConsoleTextAttribute(cmd, 12);
                        printf("%.2f\n", w);
                        SetConsoleTextAttribute(cmd, normalColor);
                    }
                    else
                        printf("%.2f\n", w);
                }
            printf("\n");
        }
    }
    SetConsoleTextAttribute(cmd, 7);
    printf("-----------------------------------------------\n\n");
}

float seayon::model::print(const dataset& data, const int sample)
{
    pulse(data[sample].x.data());
    print();

    return evaluate(data);
}

void seayon::model::printo(const bool oneline)
{
    if (oneline)
    {
        const int lastn = layers.back().nCount - 1;

        printf("[");
        for (int i = 0; i < lastn; ++i)
        {
            printf("%.5f, ", layers.back().neurons[i]);
        }
        printf("%.5f", layers.back().neurons[lastn]);
        printf("]\n");
    }
    else
    {
        HANDLE cmd = GetStdHandle(STD_OUTPUT_HANDLE);

        int normalColor = 11;

        int l = layers.size() - 1;

        SetConsoleTextAttribute(cmd, 11);
        printf("Output Layer:\n");

        size_t largest = std::max_element(layers[l].neurons.begin(), layers[l].neurons.end()) - layers[l].neurons.begin();
        for (int n = 0; n < layers[l].nCount; ++n)
        {
            printf("       Neuron[%02i]   ", n);
            if (l == layers.size() - 1)
            {
                if (n == largest)
                    SetConsoleTextAttribute(cmd, 95);
                else
                    SetConsoleTextAttribute(cmd, 7);
                printf("%.2f", layers[l].neurons[n]);
                SetConsoleTextAttribute(cmd, normalColor);
            }
            else
                printf("%.2f", layers[l].neurons[n]);

            if (l > 0)
            {
                if (layers[l].biases[n] <= 0.0f)
                {
                    printf("\t(");
                    SetConsoleTextAttribute(cmd, 12);
                    printf("%0.2f", layers[l].biases[n]);
                    SetConsoleTextAttribute(cmd, normalColor);
                    printf(")\n");
                }
                else
                    printf("\t(%0.2f)\n", layers[l].biases[n]);
            }
            else
                printf("\n\n");
        }
        SetConsoleTextAttribute(cmd, 7);
        printf("-----------------------------------------------\n\n");
    }
}

float seayon::model::printo(const dataset& data, const int sample, const bool oneline)
{
    pulse(data[sample].x.data());
    printo(oneline);

    return evaluate(data);
}