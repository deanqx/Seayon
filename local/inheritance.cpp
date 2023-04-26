#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
// #include "C:\Users\dean\Git\Seayon\local\B.cpp"

template <int INPUTS, int OUTPUTS>
struct trainingdata
{
    struct sample
    {
        float inputs[INPUTS]{};
        float outputs[OUTPUTS]{};
    };
};

template <int LAYERS>
class B
{
public:
    template <int INPUTS, int OUTPUTS>
    float pulse(typename trainingdata<INPUTS, OUTPUTS>::sample& sample)
    {
        return sample.inputs[INPUTS - 1] + sample.outputs[OUTPUTS - 1] + 1.0f;
    }
};

template <int LAYERS>
class cuda: public B<LAYERS>
{
public:
    using B<LAYERS>::pulse;

    template <int INPUTS, int OUTPUTS>
    float foo()
    {
        cuda<LAYERS> net;
        typename trainingdata<INPUTS, OUTPUTS>::sample s;

        return net.template pulse<INPUTS, OUTPUTS>(s);
    }
};

struct layer
{
    const int a;

    layer(): a(0)
    {
    }

    layer(const int& a): a(a + 1)
    {
    }
};

class MyParent
{
public:
    struct Sub
    {
        int x{};
    };
};

class MyChild: public MyParent
{
public:
    using MyParent::Sub;

    struct Extra
    {
        struct Sub
        {
            int y{};
        };

        std::vector<MyChild::Sub> parent_sub;

        int boo()
        {
            parent_sub.emplace_back();
            return parent_sub[0].x;
        }
    };

    int foo()
    {
        Extra extra;
        return extra.boo();
    }
};

int main()
{
    MyChild my;
    my.foo();

    printf("hello\b");
    std::cin.get();

    cuda<4> c;
    // printf("%f\n", c.foo<10, 10>());
}