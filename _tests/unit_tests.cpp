#include "gtest/gtest.h"
#include "seayon.hpp"

namespace
{
    using namespace seayon;

    static dataset<2, 2> data =
    {
        {{1.0f, 0.0f}, {0.0f, 1.0f}},
         {{0.0f, 1.0f}, {1.0f, 0.0f}}
    };

    static std::vector<int> layout = { 2, 3, 4, 2 };
    static std::vector<ActivFunc> funcs = { ActivFunc::LEAKYRELU, ActivFunc::LEAKYRELU, ActivFunc::SIGMOID };

    float pulse()
    {
        model m(layout, funcs, 1472, false);

        m.pulse(data[0].inputs, 2);

        return m.layers[3].neurons[0] + m.layers[3].neurons[1];
    }
    float fit()
    {
        model m(layout, funcs, 1472, false);

        float sum = 0.0f;

        // TODO test batch_size

        m.fit(20, data, data, Optimizer::STOCHASTIC, 0.5f, 0.5f);
        m.pulse(data[0].inputs, 2);
        sum += m.layers[3].neurons[0] + m.layers[3].neurons[1];

        return sum;
    }

    float accruacy()
    {
        model m(layout, funcs, 1472, false);

        return m.accruacy(data);
    }
    float loss()
    {
        model m(layout, funcs, 1472, false);

        return m.loss(data);
    }

    bool equals()
    {
        model m(layout, funcs, 1472, false);
        model m2(layout, funcs, 1471, false);

        return m.equals(m) == true && m.equals(m2) == false;
    }
    float combine()
    {
        model m(layout, funcs, 1472, false);
        model m2(layout, funcs, 1471, false);

        m.combine(&m2, 1);

        return m.layers[3].neurons[0] + m.layers[3].neurons[1];
    }
    bool copy()
    {
        model m(layout, funcs, 1472, false);
        model m2(layout, funcs, 1471, false);

        m.copy(m2);

        return m.equals(m2);
    }
    bool save_load()
    {
        model m(layout, funcs, 1472, false);

        std::vector<char> buffer;
        m.save(buffer);

        model_parameters para;
        para.load_parameters(buffer.data());

        model m2(para);
        m2.load(buffer.data());

        return m.equals(m2);
    }
}

TEST(Basis, Activation)
{
    EXPECT_EQ(pulse(), 1.01069021f);
}
TEST(Basis, Training)
{
    EXPECT_EQ(fit(), 1.01121223f);
}

TEST(Analysis, Accruacy)
{
    EXPECT_EQ(accruacy(), 0.5f);
}
TEST(Analysis, Loss)
{
    EXPECT_EQ(loss(), 0.267207265f);
}

TEST(Management, Equals)
{
    EXPECT_EQ(equals(), true);
}
TEST(Management, Combine)
{
    EXPECT_EQ(combine(), 0.0f);
}
TEST(Management, Copy)
{
    EXPECT_EQ(copy(), true);
}
TEST(Management, SaveAndLoad)
{
    EXPECT_EQ(save_load(), true);
}