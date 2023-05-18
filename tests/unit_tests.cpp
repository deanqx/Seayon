#include "gtest/gtest.h"
#include "seayon.hpp"

namespace
{
    using namespace seayon;

    static seayon::dataset data(
        std::vector<std::vector<float>>{ {1.0f, 0.0f}, { 0.0f, 1.0f } },
        std::vector<std::vector<float>>{ {0.0f, 1.0f}, { 1.0f, 0.0f } }
    );

    static std::vector<int> layout = { 2, 3, 4, 2 };
    static std::vector<ActivFunc> funcs = { ActivFunc::RELU, ActivFunc::LEAKYRELU, ActivFunc::SIGMOID };

    float pulse()
    {
        printf("--- %i ---\n", data.size());
        printf("--- %f, %f ---\n", data[0].x[0], data[0].x[1]);
        model m(layout, funcs, 1472, false);

        m.pulse(data[0].x);

        return m.layers[3].neurons[0] + m.layers[3].neurons[1];
    }
    float fit()
    {
        model m(layout, funcs, 1472, false);

        m.fit(data, data, false, 20, 1, 2, 0.1f);

        float x = m.loss(data);

        return x;
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

        model* array[]{ &m, &m2 };
        m.combine_into(array, 2);

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
    EXPECT_EQ(pulse(), 1.01520085f);
}
TEST(Basis, Training)
{
    const float ret = fit();
    printf("Return: %f\n", ret);

    EXPECT_TRUE(ret < 0.01f);
}

TEST(Analysis, Accruacy)
{
    EXPECT_EQ(accruacy(), 0.5f);
}
TEST(Analysis, Loss)
{
    EXPECT_EQ(loss(), 0.268624932f);
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