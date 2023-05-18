#ifndef SEAYON_HPP
#define SEAYON_HPP

#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <memory>

#ifndef NOMINMAX
#define NOMINMAX
#endif

namespace seayon
{
    float Sigmoid(const float z);
    float dSigmoid(const float a);
    float Tanh(const float z);
    float dTanh(const float a);
    float ReLu(const float z);
    float dReLu(const float a);
    float LeakyReLu(const float z);
    float dLeakyReLu(const float a);

    typedef float(*ActivFunc_t)(const float);
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
            float* const x;
            float* const y;

            sample(const int inputSize, const int outputSize) : x(new float[inputSize]), y(new float[outputSize])
            {}
            sample(float* const inputs, float* const outputs) : x(inputs), y(outputs)
            {}

            void clear();
        };

    private:
        sample* samples = nullptr;
        int sampleCount = 0;
        const bool manageMemory;

    public:
        const int xsize;
        const int ysize;

        dataset(const int inputSize, const int outputSize, bool manageMemory = false) : xsize(inputSize), ysize(outputSize), manageMemory(manageMemory)
        {}
        dataset(const int inputSize, const int outputSize, const int reserved) : xsize(inputSize), ysize(outputSize), manageMemory(true)
        {
            reserve(reserved);
        }
        /**
         * Introduced for cuda
         * @param manageMemory When enabled sample array will be deleted
         */
        dataset(const int inputSize, const int outputSize, sample* samples, const int sampleCount, const bool manageMemory)
            : samples(samples), sampleCount(sampleCount), manageMemory(manageMemory), xsize(inputSize), ysize(outputSize)
        {}
        dataset(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& outputs);
        /**
         * Allocates new memory without clearing it (size() is updated)
         * @param reserved New sample count
         */
        void reserve(const int reserved);
        /**
         * Introduced for cuda
         * @param manageMemory When enabled sample array will be deleted
         */
        int size() const;
        sample& operator[](const int i) const;
        sample* get(const int i) const;
        size_t save(std::vector<char>& out_buffer);
        size_t save(std::ofstream& file);
        void load(const char* buffer);
        bool load(std::ifstream& file);
        // has to be same constructor values
        void swap(dataset* with);
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
        void normalize(const float max, const float min);
        /**
         * Randomizes order of samples
         */
        void shuffle();
        void clear();
        ~dataset();
    };

    struct model_parameters
    {
        bool printloss{};
        int seed{};
        std::vector<int> layout{};
        std::vector<ActivFunc> a{};
        std::string logfolder{};

        void load_parameters(const char* buffer);
        bool load_parameters(std::ifstream& file);
    };

    // Open source Neural Network library in C++
    class model
    {
    public:
        struct layer
        {
        protected:
            const bool manageMemory;
        public:
            const ActivFunc func;
            ActivFunc_t activation;
            ActivFunc_t derivative;

            const int nCount;
            const int wCount;

            /**
            * Goes from second to first
            * @tparam layers[l2].weights [n2 * n1Count + n1]
            */
            float* const weights;
            float* const neurons;
            float* const biases;

            // layer(): manageMemory(false), func(ActivFunc::SIGMOID), nCount(0), wCount(0), neurons(nullptr), biases(nullptr), weights(nullptr)
            // {
            // }

            layer(const ActivFunc func,
                ActivFunc_t activation,
                ActivFunc_t derivative,
                float* const neurons,
                float* const biases,
                float* const weights,
                const int nCount,
                const int wCount,
                const bool manageMemory) :
                func(func),
                activation(activation),
                derivative(derivative),
                neurons(neurons),
                biases(biases),
                weights(weights),
                nCount(nCount),
                wCount(wCount),
                manageMemory(manageMemory)
            {}

            layer(const int PREVIOUS, const int NEURONS, const ActivFunc func);
            ~layer();
        };

        const bool manageMemory;
        const int seed;
        const bool printloss;
        const std::string logfolder;

        const int xsize;
        const int ysize;

        const int layerCount;
        layer* const layers;

        model(const bool manageMemory,
            const int seed,
            const bool printloss,
            const std::string logfolder,
            const int logLenght,
            const int xsize,
            const int ysize,
            const int layerCount,
            layer* const layers,
            const float* const* inl,
            const float* const* outl) :
            manageMemory(manageMemory),
            seed(seed),
            printloss(printloss),
            logfolder(logfolder),
            xsize(xsize),
            ysize(ysize),
            layerCount(layerCount),
            layers(layers)
        {}

        /**
         * Creates network where every neuron is connected to each neuron in the next layer.
         * @param layout Starts with the input layer (Minimum 2 layers)
         * @param a Activation function for each layer (first one will be ignored)
         * @param seed Random weight seed (-1 generates random seed by time)
         * @param printloss Print loss() value while training (every second, high performance consumption)
         * @param logfolder Write log file for training progress (keep empty to disable)
         */
        model(const std::vector<int> layout, const std::vector<ActivFunc> a, int seed = -1, bool printloss = true, std::string logfolder = std::string());
        model(const model_parameters& para) : model(para.layout, para.a, para.seed, para.printloss, para.logfolder)
        {}
        ~model();
        void whatsetup();

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
        size_t save(std::ofstream& file) const;
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
        bool load(std::ifstream& file);
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
        std::vector<float> denormalized(const float max, const float min) const;
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
        void printo();
        /**
         * Prints only the output layer with the loss() and the accruacy()
         * @return difference value "diff()"
         */
        float printo(const dataset& data, const int sample);
        // Prints output layer in one line. pulse() should be called before
        void print_one();
        /**
         * Prints output layer in one line
         * @return difference value "diff()"
         */
        float print_one(const dataset& data, const int sample);
        /**
         * Trains the network with Gradient Descent to minimize the loss function (you can cancel with 'q')
         * @param max_iterations Begin small
         * @param traindata The large dataset
         * @param testdata The small dataset which the network never saw before
         * @param learningRate Lower values generate more reliable but also slower results
         * @param momentum Can accelerate training but also produce worse results (disable with 0.0f)
         * @param total_threads aka batch size divides training data into chunks to improve performance for large networks (not used by stochastic g.d.)
         */
        void fit(const dataset& traindata, const dataset& testdata, const bool shuffle,
            int max_epochs = 1, int batch_size = 1, int thread_count = 1, float learning_rate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-7f);

    protected:
        bool check(const dataset& data) const;

        class fitlog
        {
            model& parent;
            const int sampleCount;
            const dataset& testdata;
            const int max_iterations;
            const bool printloss;

            std::unique_ptr<std::ofstream> file{};
            size_t lastLogLenght = 0;
            int lastLogAt = 0;
            std::chrono::high_resolution_clock::time_point overall;
            std::chrono::high_resolution_clock::time_point sampleTimeLast;
            std::chrono::high_resolution_clock::time_point last;

            float lastLoss[5]{};
            int lastLossIndex = 0;

        public:
            float log(int epoch);
            fitlog(model& parent, const int& sampleCount, const dataset& testdata, const int& max_iterations, const bool& printloss, const std::string& logfolder);
        };
    };
}

#include "src/activation.cpp"
#include "src/basis.cpp"
#include "src/dataset.cpp"
#include "src/evaluation.cpp"
#include "src/fitlog.cpp"
#include "src/model_para.cpp"
#include "src/model.cpp"
#include "src/print.cpp"
#include "src/training.cpp"

#endif