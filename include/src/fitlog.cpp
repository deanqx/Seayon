#include "../seayon.hpp"
#include <sstream>
#include <iomanip>
#include <iostream>

#ifdef _WIN32
#include <conio.h>
#endif

void resolveTime(long long seconds, int* resolved)
{
    resolved[0] = (int)(seconds / 3600LL);
    seconds -= 3600LL * (long long)resolved[0];
    resolved[1] = (int)(seconds / 60LL);
    seconds -= 60LL * (long long)resolved[1];
    resolved[2] = (int)(seconds);
}

float seayon::model::fitlog::log(int epoch)
{
    float l = -1.0f;

    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds sampleTime = std::chrono::duration_cast<std::chrono::microseconds>(now - sampleTimeLast);
    if (sampleTime.count() > 1000000LL || epoch == max_iterations || epoch == 0)
    {
        sampleTimeLast = now;
        std::chrono::milliseconds elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last);
        std::chrono::seconds runtime = std::chrono::duration_cast<std::chrono::seconds>(now - overall);

        if (printloss)
            l = parent.loss(testdata);

        float progress = (float)epoch * 100.0f / (float)max_iterations;

        int samplesPerSecond = 0;
        if (epoch > 0)
        {
            if (epoch > lastLogAt)
                sampleTime /= epoch - lastLogAt;
            if (sampleTime.count() < 1)
                samplesPerSecond = -1;
            else
                samplesPerSecond = (int)((int64_t)sampleCount * 1000LL / sampleTime.count());
        }

        int runtimeResolved[3];
        resolveTime(runtime.count(), runtimeResolved);

        if (epoch > lastLogAt)
            elapsed /= epoch - lastLogAt;
        elapsed *= max_iterations - epoch;

        std::chrono::seconds eta = std::chrono::duration_cast<std::chrono::seconds>(elapsed);

        int etaResolved[3];
        resolveTime(eta.count(), etaResolved);

        std::ostringstream message;
        message << epoch << "/" << max_iterations << std::setw(9)
            << samplesPerSecond << "k Samples/s " << std::setw(13)
            << "Runtime: " << runtimeResolved[0] << "h " << runtimeResolved[1] << "m " << runtimeResolved[2] << "s " << std::setw(9)
            << "ETA: " << etaResolved[0] << "h " << etaResolved[1] << "m " << etaResolved[2] << "s" << std::setw(9);

        if (l > -1.0f)
            message << "loss: " << std::setprecision(2) << std::defaultfloat << l;

        const int cleared = std::max(0, (int)lastLogLenght - (int)message.str().length());
        std::cout << std::string(lastLogLenght, '\b') << message.str() << std::string(cleared, ' ');
        lastLogLenght = message.str().length() + cleared;

        if (file.get() != nullptr)
            *file << samplesPerSecond << ',' << runtime.count() << ',' << eta.count() << ',' << l << '\n';

        if (lastLoss[0] <= l
            && lastLoss[1] <= l
            && lastLoss[2] <= l
            && lastLoss[3] <= l
            && lastLoss[4] <= l && epoch > 10 || kbhit() && getch() == 'q')
        {
            return 0.0f;
        }
        lastLoss[lastLossIndex++] = l;
        if (lastLossIndex == 5)
            lastLossIndex = 0;

        lastLogAt = epoch;
        last = std::chrono::high_resolution_clock::now();
    }

    return l;
}

seayon::model::fitlog::fitlog(model& parent, const int& sampleCount, const dataset& testdata, const int& max_iterations, const bool& printloss, const std::string& logfolder) :
    parent(parent), sampleCount(sampleCount), testdata(testdata), max_iterations(max_iterations), printloss(printloss)
{
    if (!logfolder.empty())
    {
        std::string path(logfolder + "log.csv");
        std::ifstream exists(path);
        for (int i = 1; i < 16; ++i)
        {
            if (!exists.good())
            {
                break;
            }
            exists.close();
            path = logfolder + "log(" + std::to_string(i) + ").csv";
            exists.open(path);
        }

        file = std::make_unique<std::ofstream>(path);
        *file << "SamplesPer(seconds),Runtime(seconds),ETA(seconds),loss" << std::endl;
    }

    printf("\n");
    overall = std::chrono::high_resolution_clock::now();
    last = overall;
    sampleTimeLast = overall;
    lastLoss[lastLossIndex++] = 999999.0f;
    log(0);
}