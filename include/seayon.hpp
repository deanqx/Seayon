#ifndef SEAYON_HPP
#define SEAYON_HPP

#include <vector>
#include <string>
#include <chrono>
#include <memory>
#include <random>

#ifndef NOMINMAX
#define NOMINMAX
#endif

namespace seayon
{
    float Sigmoid(float z);
    float dSigmoid(float a);
    float Tanh(float z);
    float dTanh(float a);
    float ReLu(float z);
    float dReLu(float a);
    float LeakyReLu(float z);
    float dLeakyReLu(float a);

    typedef float(*ActivFunc_t)(float);
    enum class ActivFunc
    {
        LINEAR,
        SIGMOID,
        TANH,
        RELU,
        LEAKYRELU
    };

    // Stores and manages dataset in memory
    struct dataset
    {
        struct sample
        {
            std::vector<float> x;
            std::vector<float> y;

            sample(const int inputSize, const int outputSize)
            {
                x.resize(inputSize);
                y.resize(outputSize);
            }
            sample(std::vector<float>& inputs, std::vector<float>& outputs)
            {
                x.swap(inputs);
                y.swap(outputs);
            }
        };

        std::vector<sample> samples;
        const int xsize;
        const int ysize;

        dataset(int inputSize, int outputSize);
        dataset(int inputSize, int outputSize, int newsize);
        dataset(std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& outputs);
        void resize(int newsize);
        inline const sample& operator[](const int index) const;
        size_t save(std::vector<char>& out_buffer) const;
        size_t save_file(const char* path) const;
        void load(const char* buffer);
        bool load_file(const char* path);
        /**
         * @return Highest value in dataset
         */
        float max_value() const;
        /**
         * @return Lowest value in dataset
         */
        float min_value() const;
        /**
         * Normalizes all values between max and min
         * @param max Highest value in dataset (use this->max_value())
         * @param min Lowest value in dataset (use this->min_value())
         */
        void normalize(float max, float min);
        std::vector<sample> denormalized(float max, float min) const;
        /**
         * Randomizes order of samples (slow should not be used regularly)
         */
        void shuffle(int seed);
        void combine(dataset& from);
        void split(dataset& into, float splitoff);
    };

    struct model_parameters
    {
        int seed{};
        std::vector<int> layout{};
        std::vector<ActivFunc> a{};
        std::string logfolder{};

        void load_parameters(const char* buffer);
        bool load_parameters_file(const char* path);
    };

    // Open source Neural Network library in C++
    class model
    {
    public:
        struct layer
        {
            const ActivFunc func;
            ActivFunc_t activation;
            ActivFunc_t derivative;

            const int nCount;
            const int wCount;

            /**
            * Goes from second to first
            * @tparam layers[l2].weights [n2 * n1Count + n1]
            */
            std::vector<float> neurons;
            std::vector<float> biases;
            std::vector<float> weights;

            layer(int PREVIOUS, int NEURONS, ActivFunc func, std::mt19937& gen);
        };

        int seed;
        const std::string logfolder;

        const int xsize;
        const int ysize;

        std::vector<layer> layers;

        /**
         * Creates network where every neuron is connected to each neuron in the next layer.
         * @param layout Starts with the input layer (Minimum 2 layers)
         * @param a Activation function for each layer (first one will be ignored)
         * @param seed Random weight seed (-1 generates random seed by time)
         * @param printloss Print loss() value while training (every second, high performance consumption)
         * @param logfolder Write log file for training progress (keep empty to disable)
         */
        model(const std::vector<int> layout, const std::vector<ActivFunc> a, int seed = -1, std::string logfolder = std::string());
        model(const model_parameters& para);

        /**
         * Stores all weights and biases in one binary buffer
         * @return size of buffer
         */
        size_t save(std::vector<char>& buffer) const;
        /**
         * Stores all weights and biases in one binary file
         * @param file use std::ios::binary
         * @return size of buffer
         */
        size_t save_file(const char* path) const;
        /**
         * Loads binary network buffer
         * @exception Currupt data can through an error
         */
        void load(const char* buffer);
        /**
         * Loads binary network file
         * @param file use std::ios::binary
         * @exception Currupt files can through an error
         * @return if successful
         */
        bool load_file(const char* path);
        // Copies weights and biases to a different instance
        void copy(model& to) const;
        /**
         * Combines array of networks by averaging the values. And overwrites current main
         * @param with Array of pointers to networks
         * @param count How many networks
         */
        void combine_into(model** with, int count);
        /**
         * Compares weights and biases
         * @return true: equal; false: not equal
         */
        bool equals(model& second);
        /**
         * Transforms normalized output back to original scale
         * @param max Highest value in dataset (use this->max())
         * @param min Lowest value in dataset (use this->max())
         * @return Denormalized output layer
         */
        void denormalize(float max, float min);
        /**
         * Calculates network's outputs (aka predict)
         * @return Pointer to output layer/array
         */
        float* pulse(const float* inputs);
        /**
         * sum of (1 / PARAMETERS) * (x - OPTIMAL)^2
         * @param sample Optimal outputs (testdata)
         * @return lower means better
         */
        float loss(const typename dataset::sample& sample);
        /**
         * sum of (1 / PARAMETERS) * (x - OPTIMAL)^2
         * @param data Optimal outputs (testdata)
         * @return lower means better
         */
        float loss(const dataset& data);
        /**
         * sum of (1 / PARAMETERS) * abs(x - OPTIMAL)
         * @param sample Optimal outputs (testdata)
         * @return lower means better
         */
        float diff(const typename dataset::sample& sample, std::vector<float> factor);
        /**
         * sum of (1 / PARAMETERS) * abs(x - OPTIMAL)
         * @param data Optimal outputs (testdata)
         * @return lower means better
         */
        float diff(const dataset& data, std::vector<std::vector<float>> factors = std::vector<std::vector<float>>());
        /**
         * sum of (1 / PARAMETERS) * abs(x - OPTIMAL)
         * @param sample Optimal outputs (testdata)
         * @return lower means better
         */
        float diff_max(const typename dataset::sample& sample, std::vector<float> factor);
        /**
         * sum of (1 / PARAMETERS) * abs(x - OPTIMAL)
         * @param data Optimal outputs (testdata)
         * @return lower means better
         */
        float diff_max(const dataset& data, std::vector<std::vector<float>> factors = std::vector<std::vector<float>>());
        /**
         * sum of (1 / PARAMETERS) * abs(x - OPTIMAL)
         * @param sample Optimal outputs (testdata)
         * @return lower means better
         */
        float diff_min(const typename dataset::sample& sample, std::vector<float> factor);
        /**
         * sum of (1 / PARAMETERS) * abs(x - OPTIMAL)
         * @param data Optimal outputs (testdata)
         * @return lower means better
         */
        float diff_min(const dataset& data, std::vector<std::vector<float>> factors = std::vector<std::vector<float>>());
        /**
         * for each sample: does highest output matches optimal highest
         * @param data Optimal outputs (testdata)
         * @return percentage, higher means better
         */
        float accruacy(const dataset& data);
        void whatsetup();
        /**
         * Prints all rating functions
         * @return difference value "diff()"
         */
        float evaluate(const dataset& data);
        // Prints all values. pulse() should be called before
        void print();
        /**
         * Prints all values with the loss() and the accruacy()
         * @return difference value "diff()"
         */
        float print(const dataset& data, int sample);
        // Prints only the output layer. pulse() should be called before
        void printo(bool oneline = false);
        /**
         * Prints only the output layer with the loss() and the accruacy()
         * @return difference value "diff()"
         */
        float printo(const dataset& data, int sample, bool oneline = false);

        typedef bool(*step_callback_t)(model& main, int epoch, float l, float val_l, const dataset& traindata, const dataset& testdata);
        /**
         * Trains the network with Gradient Descent to minimize the loss function (you can cancel with 'q')
         * @param max_iterations Begin small
         * @param traindata The large dataset
         * @param testdata The small dataset which the network never saw before
         * @param learningRate Lower values generate more reliable but also slower results
         * @param momentum Can accelerate training but also produce worse results (disable with 0.0f)
         * @param total_threads aka batch size divides training data into chunks to improve performance for large networks (not used by stochastic g.d.)
         */
        void fit(const dataset& traindata,
            const dataset& testdata,
            int epochs = 1,
            int batch_size = 1,
            int verbose = 2,
            bool shuffle = true,
            float steps_per_epoch = 1.0f,
            float learning_rate = 0.001f,
            std::vector<float> dropouts = std::vector<float>(),
            step_callback_t callback = nullptr,
            float beta1 = 0.9f,
            float beta2 = 0.999f,
            float epsilon = 1e-7f);

    protected:
        bool check(const dataset& data) const;
    };
}

#include "src/activation.cpp"
#include "src/basis.cpp"
#include "src/dataset.cpp"
#include "src/evaluation.cpp"
#include "src/model_para.cpp"
#include "src/model.cpp"
#include "src/print.cpp"
#include "src/training.cpp"

#endif