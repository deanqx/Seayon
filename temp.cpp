// #define RELEASE
#include "include/timez/timez.hpp"

int main()
{
    timez::init();

    int x = 0;
    int y = 0;
    for (int j = 0; j < 100; ++j)
    {
        {
            timez::perf A("Calculate: x");

            for (int i = 0; i < 1000; ++i)
            {
                timez::perf loop("loop");

                x += i;
            }
        }

        {
            timez::perf B("Calculate: y");

            {
                timez::perf B("Part 1");
                y = x * 2;
            }

            {
                timez::perf B("Part 2");
                y *= y;
            }

            {
                timez::perf B("Part 3(hidden)");
                ++y;
            }
        }
    }

    // recalcTotal: Removes inaccuracies from for example loops
    timez::print(true, std::vector<std::string>{"Part 3(hidden)"});
    timez::clean();

    return 0;
}