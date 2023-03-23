#include "seayon.hpp"

int main()
{
    trainingdata<2, 2, 2> data;
    data.samples[0].inputs[0] = 1.0f;
    data.samples[0].inputs[1] = 0.0f;
    data.samples[0].outputs[0] = 0.0f;
    data.samples[0].outputs[1] = 1.0f;
    data.samples[1].inputs[0] = 0.0f;
    data.samples[1].inputs[1] = 1.0f;
    data.samples[1].outputs[0] = 1.0f;
    data.samples[1].outputs[1] = 0.0f;

    int layout[]{ 2, 5, 2 };
    seayon<2> nn(layout, ActivFunc::SIGMOID, 1472);

    printf("%f", nn.accruacy(data));
}