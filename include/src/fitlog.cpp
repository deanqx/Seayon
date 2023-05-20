#include "../seayon.hpp"
#include <sstream>
#include <iomanip>
#include <iostream>
#include <fstream>

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

class fitlog
{
    seayon::model& parent;
    const int sampleCount;
    const seayon::dataset& traindata;
    const seayon::dataset& testdata;
    const int max_iterations;
    const bool val_loss;

    std::unique_ptr<std::ofstream> file{};
    size_t lastLogLenght = 0;
    int lastLogAt = 0;
    std::chrono::high_resolution_clock::time_point overall;
    std::chrono::high_resolution_clock::time_point sampleTimeLast;
    std::chrono::high_resolution_clock::time_point last;

public:
    bool log(int epoch, const float l1)
    {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::microseconds sampleTime = std::chrono::duration_cast<std::chrono::microseconds>(now - sampleTimeLast);
        if (sampleTime.count() > 1000000LL || epoch == max_iterations || epoch == 0)
        {
            sampleTimeLast = now;
            std::chrono::milliseconds elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last);
            std::chrono::seconds runtime = std::chrono::duration_cast<std::chrono::seconds>(now - overall);

            float l2 = -1.0f;

            if (val_loss)
            {
                l2 = parent.loss(testdata);
            }

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
                << "ETA: " << etaResolved[0] << "h " << etaResolved[1] << "m " << etaResolved[2] << "s" << std::setw(9)
                << "loss: " << std::setprecision(2) << std::defaultfloat << l1;

            if (l2 > -1.0f)
                message << " val_loss: " << std::setprecision(2) << std::defaultfloat << l2;

            const int cleared = std::max(0, (int)lastLogLenght - (int)message.str().length());
            std::cout << std::string(lastLogLenght, '\b') << message.str() << std::string(cleared, ' ');
            lastLogLenght = message.str().length() + cleared;

            if (file.get() != nullptr)
                *file << epoch << ',' << samplesPerSecond << ',' << runtime.count() << ',' << eta.count() << ',' << l1 << ',' << l2 << '\n';

            if (kbhit() && getch() == 'q')
            {
                return true;
            }

            lastLogAt = epoch;
            last = std::chrono::high_resolution_clock::now();
        }

        return false;
    }
    fitlog(seayon::model& parent, const int& sampleCount, const seayon::dataset& traindata, const seayon::dataset& testdata, const int& max_iterations, const bool& val_loss, const std::string& logfolder)
        : parent(parent), sampleCount(sampleCount), traindata(traindata), testdata(testdata), max_iterations(max_iterations), val_loss(val_loss)
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
            *file << "epoch,SamplesPer(seconds),Runtime(seconds),ETA(seconds),loss,val_loss" << std::endl;
        }

        printf("\n");
        overall = std::chrono::high_resolution_clock::now();
        last = overall;
        sampleTimeLast = overall;
        log(0, parent.loss(traindata));
    }
};