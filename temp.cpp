#define LAYERS 3

struct layer
{
    const static int nCount = 10;
    const static int wCount = 20;
    float neurons[nCount]{};
    float weights[wCount]{};
};

inline float dSigmoid(float a)
{
    return a * (1.0f - a);
}

int main()
{
    layer layers[LAYERS];

    const int lastl = LAYERS - 1;

    float* dn[LAYERS];
    float* lastdb[LAYERS];
    float* lastdw[LAYERS];

    for (int l = 0; l < LAYERS; ++l)
    {
        dn[l] = new float[layers[l].nCount];
        lastdb[l] = new float[layers[l].nCount]();
        lastdw[l] = new float[layers[l].wCount]();
    }

    for (int n2 = 0; n2 < layers[lastl].nCount; ++n2)
    {
        dn[lastl][n2] = dSigmoid(layers[lastl].neurons[n2]) * 2 * (layers[lastl].neurons[n2] - 1);
    }

    for (int l2 = lastl; l2 >= 2; --l2)
    {
        const int l1 = l2 - 1;
        const int ncount = layers[l2].nCount;

        for (int n1 = 0; n1 < layers[l1].nCount; ++n1)
        {
            const int rowindex = n1 * ncount;

            float error = 0;
            for (int n2 = 0; n2 < ncount; ++n2)
                error += dn[l2][n2] * layers[l2].weights[n2 + rowindex];

            dn[l1][n1] = dSigmoid(layers[l1].neurons[n1]) * error;
        }
    }

    for (int l2 = lastl; l2 >= 1; --l2)
    {
        const int l1 = l2 - 1;
        const int& ncount = layers[l2].nCount;

        for (int n1 = 0; n1 < layers[l1].nCount; ++n1)
        {
            const int rowindex = n1 * ncount;
            for (int n2 = 0; n2 < ncount; ++n2)
            {
                const int windex = n2 + rowindex;
                const float dw = layers[l1].neurons[n1] * -dn[l2][n2];
                layers[l2].weights[windex] += dw + lastdw[l2][windex];
                lastdw[l2][windex] = dw;
            }
        }
    }
}