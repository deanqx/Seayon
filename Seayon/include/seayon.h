#pragma once

#include <vector>
#include <math.h>
#include <fstream>

inline float ReLu(float z)
{
	return (z < 0.0f ? 0.0f : z);
}
inline float dReLu(float a)
{
	return (a < 0.0f ? 0.0f : 1.0f);
}
inline float Sigmoid(float z)
{
	return 1.0f / (1.0f + exp(-z));
}
inline float dSigmoid(float a)
{
	return a * (1.0f - a);
}

// TODO Rewrite Disciptions
// Open source Neural Network library in C++ with lots of easy to use features. Copyright by Dean Schneider (deanqx, Sawey)
class seayon
{
public:
	struct Layer
	{
		std::vector<float> Neurons; // TODO Remove vector
		/**
		 * Goes from first to second
		 * @tparam Layers[l2].Weights[n2][n1]
		 */
		std::vector<std::vector<float>> Weights;
		std::vector<float> Biases;
	};
	std::vector<Layer> Layers;

	struct trainingdata
	{
		struct sample
		{
			std::vector<float> inputs;
			std::vector<float> outputs;
		};
		std::vector<sample> samples;

		// Returns false if training data is currupt (quickcheck)
		bool check(seayon& s);
	};

	enum class ActivFunc
	{
		SIGMOID,
		RELU
	} Activation = ActivFunc::SIGMOID;

	/**
	 * Creates network where every neuron is connected to each neuron in the next layer.
	 * @param layerCount Starts with the input layer (Minimum 2 layers)
	 * @param ActivFunc Activation function for all neurons.
	 */
	void generate(const std::vector<int> layerCounts, const ActivFunc a = ActivFunc::SIGMOID, const int seed = -1);
	// Saves network to a .nn file
	void save(std::ofstream& file);
	// Copys network to a std::string
	std::string save();
	/**
	 * Loads .nn file
	 * @exception Currupt .nn files will throw an error!
	 */
	void load(std::ifstream& file);
	/**
	 * Loads network from a std::string
	 * @exception Currupt string will throw an error!
	 */
	void load(std::string s);
	void copy(seayon& to);
	/**
	 * Combines two networks with the average values.
	 * @param with List of networks
	 * @param count How many networks
	 */
	void combine(seayon* with, size_t count);
	bool equals(seayon& second);

	// Prints all values. pulse() should be called before
	void print();
	// Prints all values with the cost() and the accruacy().
	void print(trainingdata& data, int sample);
	// Prints out the output layer. pulse() should be called before
	void printo();
	// Prints out the output layer with the cost() and the accruacy().
	void printo(trainingdata& data, int sample);

	// Calculates network outputs
	void pulse(trainingdata::sample& sample);
	/**
	 * Describes how far apart the optimal outputs from the outputs are.
	 * @param outputs This are the optimal outputs
	 * @return 0.0 = Best; 1.0 or more = Bad
	 */
	float cost(trainingdata::sample& sample);
	/**
	 * Describes how far apart the optimal outputs from the outputs are.
	 * @param outputs This are the optimal outputs
	 * @return 0.0 = Best; 1.0 or more = Bad
	 */
	float cost(trainingdata& data);
	/**
	 * Describes how often the network would choose the right neurons to fire (neuron value >= 0.5).
	 * @param inputs Set values of the input layer
	 * @param outputs This are the optimal outputs
	 * @return 1.0 = 100%; 0.5f = 50%; 0.0 = 0%
	 */
	float accruacy(trainingdata& data);

	/**
	 * Backpropagation is an efficient training algorithm which works for many cases.
	 * @param runCount How many iterations. Tip: Start small with like 5 - 20
	 * @param print Writes detailed information about the progress to the console.
	 * @param learningRate Large numbers will take less runs but can "over shoot" the right value.
	 * @param momentum Large numbers will take less runs but can "over shoot" the right value.
	 */
	void fit(trainingdata& data, int runCount, const bool print, std::ofstream* logfile, float learningRate = 0.03f, float momentum = 0.1f);
};