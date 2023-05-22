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
    const int epochs;
    const bool val_loss;
    const int verbose;
    seayon::model::step_callback_t callback;

    std::unique_ptr<std::ofstream> file{};
    size_t lastLogLenght = 0;
    int lastLogAt = 0;
    std::chrono::high_resolution_clock::time_point overall;
    std::chrono::high_resolution_clock::time_point sampleTimeLast;
    std::chrono::high_resolution_clock::time_point last;

public:
    bool log(int epoch, const float l1)
    {
        float l2 = -1.0f;

        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::microseconds sampleTime = std::chrono::duration_cast<std::chrono::microseconds>(now - sampleTimeLast);
        if (callback || verbose > 2 || verbose != 0 && (sampleTime.count() > 1000000LL || epoch == epochs || epoch == 0))
        {
            sampleTimeLast = now;
            std::chrono::milliseconds elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last);
            std::chrono::seconds runtime = std::chrono::duration_cast<std::chrono::seconds>(now - overall);

            if (val_loss)
            {
                l2 = parent.loss(testdata);
            }

            float progress = (float)epoch * 100.0f / (float)epochs;

            std::string unit = "us/step";
            unsigned time_per_step = 0;
            unsigned usPerStep = 0;
            if (epoch > 0)
            {
                time_per_step = (unsigned)(sampleTime.count() / (int64_t)sampleCount);
                usPerStep = time_per_step;

                if (time_per_step > 1000)
                {
                    unit = "ms/step";
                    time_per_step /= 1000;

                    if (time_per_step > 1000)
                    {
                        unit = "s/step";
                        time_per_step /= 1000;
                    }
                }
            }

            int runtimeResolved[3];
            resolveTime(runtime.count(), runtimeResolved);

            if (epoch > lastLogAt)
                elapsed /= epoch - lastLogAt;
            elapsed *= epochs - epoch;

            std::chrono::seconds eta = std::chrono::duration_cast<std::chrono::seconds>(elapsed);

            int etaResolved[3];
            resolveTime(eta.count(), etaResolved);

            std::ostringstream runtime_str;
            std::ostringstream eta_str;
            std::ostringstream loss_str;
            runtime_str << "   Runtime: " << runtimeResolved[0] << "h " << runtimeResolved[1] << "m " << runtimeResolved[2] << "s";
            eta_str << "ETA: " << etaResolved[0] << "h " << etaResolved[1] << "m " << etaResolved[2] << "s";
            if (l1 != 0.0f)
                loss_str << "loss: " << std::setprecision(2) << std::defaultfloat << l1;

            std::ostringstream message;
            message << std::setw(3) << std::right << epoch << "/" << epochs
                << std::setw(7) << std::right << time_per_step << unit
                << std::setw(25) << std::left << runtime_str.str()
                << std::setw(18) << std::left << eta_str.str()
                << std::setw(15) << std::left << loss_str.str();

            if (l2 > -1.0f)
            {
                std::ostringstream loss2_str;
                loss2_str << "val_loss: " << std::setprecision(2) << std::defaultfloat << l2;

                message << std::setw(19) << std::left << loss2_str.str();
            }

            if (verbose > 2)
            {
                std::cout << message.str() << '\n';
            }
            else
            {
                const int clear = std::max(0, (int)lastLogLenght - (int)message.str().length());
                std::cout << std::string(lastLogLenght, '\b') << message.str() << std::string(clear, ' ');
                lastLogLenght = message.str().length() + clear;
            }

            if (file.get() != nullptr)
            {
                if (l1 != 0.0f)
                    *file << usPerStep << ',' << runtime.count() << ',' << eta.count() << ',' << l1 << ',' << l2 << '\n';
                else
                    *file << usPerStep << ',' << runtime.count() << ',' << eta.count() << ",," << l2 << '\n';
                file->flush();
            }

            if (kbhit() && getch() == 'q')
            {
                return true;
            }

            if (callback)
            {
                if (callback(parent, epoch, l1, l2, traindata, testdata))
                    return true;
            }

            lastLogAt = epoch;
            last = std::chrono::high_resolution_clock::now();
        }
        else
        {
            if (file.get() != nullptr)
            {
                if (l1 != 0.0f)
                    *file << ",,," << l1 << ",\n";
                else
                    *file << ",,,,\n";
                file->flush();
            }
        }

        return false;
    }
    fitlog(seayon::model& parent,
        const int& sampleCount,
        const seayon::dataset& traindata,
        const seayon::dataset& testdata,
        const int& epochs,
        const bool& val_loss,
        const int& verbose,
        const std::string& logfolder,
        seayon::model::step_callback_t callback)
        : parent(parent),
        sampleCount(sampleCount),
        traindata(traindata),
        testdata(testdata),
        epochs(epochs),
        val_loss(val_loss),
        verbose(verbose),
        callback(callback)
    {
        if (!logfolder.empty())
        {
            std::string path(logfolder + "log.csv");
            std::ifstream exists(path);
            for (int i = 1; i < 256; ++i)
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
            *file << "usPerStep,Runtime(sec),ETA(sec),loss,val_loss" << std::endl;
        }

        printf("\n");
        overall = std::chrono::high_resolution_clock::now();
        last = overall;
        sampleTimeLast = overall;
        log(0, 0.0f);
    }
};