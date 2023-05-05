#ifdef __cplusplus
extern "C"
{
#endif

#if(defined INPUTS && defined OUTPUTS)

#include "seayon.hpp"

    float Sigmoid(const float& z);
    float dSigmoid(const float& a);
    float Tanh(const float& z);
    float dTanh(const float& a);
    float ReLu(const float& z);
    float dReLu(const float& a);
    float LeakyReLu(const float& z);
    float dLeakyReLu(const float& a);

    typedef float(*ActivFunc_t)(const float&);

    enum class ActivFunc
    {
        LINEAR,
        SIGMOID,
        TANH,
        RELU,
        LEAKYRELU
    };

    enum class Optimizer
    {
        STOCHASTIC,
        ADAM
    };

    //...

#endif
#ifdef __cplusplus
}
#endif