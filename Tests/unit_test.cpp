#include "gtest/gtest.h"
#include "seayon.hpp"

namespace
{
    static trainingdata<2, 2> data =
    {
        {{1.0f, 0.0f}, {0.0f, 1.0f}},
         {{0.0f, 1.0f}, {1.0f, 0.0f}}
    };

    static std::vector<int> layout = { 2, 3, 4, 2 };
    static std::vector<ActivFunc> funcs = { ActivFunc::LEAKYRELU, ActivFunc::LEAKYRELU, ActivFunc::LEAKYRELU, ActivFunc::SIGMOID };

    float pulse()
    {
        seayon nn(layout, funcs, false, false, 1472, "");

        nn.pulse<2, 2>(data[0]);

        return nn[3].neurons[0] + nn[3].neurons[1];
    }
    float fit()
    {
        seayon nn(layout, funcs, false, false, 1472, "");

        float sum = 0.0f;

        nn.fit(20, data, data, Optimizer::STOCHASTIC, 0.5f, 0.5f);
        nn.pulse<2, 2>(data[0]);
        sum += nn[3].neurons[0] + nn[3].neurons[1];

        nn.fit(20, data, data, Optimizer::MINI_BATCH, 0.5f, 0.5f, 1, 1);
        nn.pulse<2, 2>(data[0]);
        sum += nn[3].neurons[0] + nn[3].neurons[1];

        return sum;
    }

    float accruacy()
    {
        seayon nn(layout, funcs, false, false, 1472, "");

        return nn.accruacy(data);
    }
    float cost()
    {
        seayon nn(layout, funcs, false, false, 1472, "");

        return nn.cost(data);
    }

    bool equals()
    {
        seayon nn(layout, funcs, false, false, 1472, "");
        seayon nn2(layout, funcs, false, false, 1471, "");

        return nn.equals(nn) == true && nn.equals(nn2) == false;
    }
    float combine()
    {
        seayon nn(layout, funcs, false, false, 1472, "");
        seayon nn2(layout, funcs, false, false, 1471, "");

        nn.combine(&nn2, 1);

        return nn[3].neurons[0] + nn[3].neurons[1];
    }
    bool copy()
    {
        seayon nn(layout, funcs, false, false, 1472, "");
        seayon nn2(layout, funcs, false, false, 1471, "");

        nn.copy(nn2);

        return nn.equals(nn2);
    }
    bool save_load()
    {
        seayon nn(layout, funcs, false, false, 1472, "");
        seayon nn2(layout, funcs, false, false, 1471, "");

        char* buffer;
        nn.save(buffer);
        nn2.load(buffer);

        delete buffer;
        return nn.equals(nn2);
    }
}

TEST(Basis, Activation)
{
    EXPECT_EQ(pulse(), 1.01069021f);
}
TEST(Basis, Training)
{
    EXPECT_EQ(fit(), 2.01150656f);
}

TEST(Analysis, Accruacy)
{
    EXPECT_EQ(accruacy(), 0.5f);
}
TEST(Analysis, Cost)
{
    EXPECT_EQ(cost(), 0.53441453f);
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