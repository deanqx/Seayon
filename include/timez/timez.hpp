#pragma once
// #define RELEASE

#include <iostream>
#include <string>
#include <chrono>
#include <algorithm>
#include <math.h>
#include <vector>
#include <sstream>

// TODO Create README with expample
class timez
{
    struct scope
    {
        scope* header;
        std::vector<scope*> sub;
        int layer = 0;

        const std::string ID = "";
        int64_t totalTime = 0;
        int64_t runs = 0;

        float percent;

        scope(const std::string ID, scope* header): ID(ID), header(header) {}
        ~scope()
        {
            for (int i = 0; i < sub.size(); ++i)
            {
                delete sub[i];
            }
        }

        void recalc()
        {
            if (sub.size() > 0)
            {
                int64_t sum = 0;
                for (scope* s : sub)
                {
                    s->recalc();
                    sum += s->totalTime;
                }
                totalTime = sum;
            }
        }
        void printp(std::vector<std::string>& hide)
        {
            if (sub.size() > 0)
            {
                std::vector<scope*> temp;
                temp.reserve(sub.size());
                for (int i = 0; i < sub.size();)
                {
                    for (int h = 0; h < hide.size(); ++h)
                    {
                        if (hide[h] == sub[i]->ID)
                        {
                            goto Continue;
                        }
                    }
                    temp.push_back(sub[i]);

                Continue:
                    ++i;
                }
                sub.swap(temp);

                int64_t overallTime = 0LL;
                for (int i = 0; i < sub.size(); ++i)
                {
                    overallTime += sub[i]->totalTime;
                }
                for (int i = 0; i < sub.size(); ++i)
                {
                    sub[i]->percent = (float)(sub[i]->totalTime * 10000LL / overallTime) * 0.01f;
                }
                std::sort(sub.begin(), sub.end(), [](const scope* a, const scope* b)
                    { return a->percent > b->percent; });

                for (int i = 0; i < sub.size(); ++i)
                {
                    std::stringstream tree;
                    for (int l = 2; l < sub[i]->layer; ++l)
                        tree << ' ';
                    if (sub[i]->layer > 1)
                        tree << (char)192;

                    tree << ceilf(sub[i]->percent * 100.0f) * 0.01f;

                    std::stringstream spaces;
                    for (int l = 1; l < sub[i]->layer; ++l)
                        spaces << "  ";
                    spaces << sub[i]->ID;

                    printf("%-15s%-15lld%-16lld%-15lld%s\n", tree.str().c_str(), sub[i]->totalTime / sub[i]->runs, sub[i]->totalTime, sub[i]->runs, spaces.str().c_str());
                    sub[i]->printp(hide);
                }
            }
        }
    };

    static scope* current; // (= Main header)

public:
    class perf
    {
        std::chrono::time_point<std::chrono::high_resolution_clock> startTimepoint;
        scope* local = nullptr;
        bool stopped = false;

    public:
        perf(const std::string id)
        {
#ifndef RELEASE
            for (int i = 0; i < current->sub.size(); ++i)
            {
                if (current->sub[i]->ID == id)
                {
                    local = current->sub[i];
                    current = local;
                    goto Continue;
                }
            }

            local = new scope(id, current);
            current = local;
            local->layer = local->header->layer + 1;
            local->header->sub.push_back(local);

        Continue:
            startTimepoint = std::chrono::high_resolution_clock::now();
#endif
        }
        ~perf()
        {
#ifndef RELEASE
            auto end = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()).time_since_epoch().count();
            auto start = std::chrono::time_point_cast<std::chrono::microseconds>(startTimepoint).time_since_epoch().count();

            if (!stopped)
            {
                stopped = true;

                local->totalTime += end - start;
                ++local->runs;
                current = local->header;
            }
#endif
        }
    };

    static void init()
    {
        current = new scope("", nullptr);
    }

    static void printp(bool recalcTotal = false, std::vector<std::string> hide = std::vector<std::string>())
    {
#ifndef RELEASE
        printf("                       ---   Performance   ---\n");
        printf("       %       Time(us)       Total(us)       Used           ID\n");

        if (recalcTotal)
            current->recalc();
        current->printp(hide);
        printf("\n");
#endif
    }

    static void clean()
    {
        // TODO
    }
};