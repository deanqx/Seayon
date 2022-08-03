#pragma once

#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <random>
#include <chrono>
#include <list>
#include <windows.h>
#include <algorithm>
#include <thread>

#define ReLu(z) (z < 0.0f ? 0.0f : z)
#define dReLu(a) (a < 0.0f ? 0.0f : 1.0f)
#define Sigmoid(z) 1.0f / (1.0f + exp(-z))
#define dSigmoid(a) a *(1.0f - a)

// Open source Neural Network library in C++ with lots of easy to use features. Copyright by deanqx and Sawey
struct seayon
{
	struct Layer
	{
		std::vector<float> Neurons;
		/**
		 * Goes from first to second
		 * @tparam Layers[l2].Weights[n2][n1]
		 */
		std::vector<std::vector<float>> Weights;
		std::vector<float> Biases;
	};
	std::vector<Layer> Layers;
	enum class ActivFunc
	{
		SIGMOID,
		RELU, 
		REMOID
	} Activation = ActivFunc::SIGMOID;
	long Writing = -1;

	/**
	 * Creates network where every neuron is connected to each neuron in the next layer.
	 * @param layerCount Starts with the input layer (Minimum 2 layers)
	 * @param ActivFunc Activation function for every neuron.
	 */
	void generate(const std::vector<int> &layerCounts, const ActivFunc a = ActivFunc::SIGMOID, const int seed = 0);
	// Saves network to a .nn file
	void save(std::ofstream &stream, const bool readFormat = false);
	// Copys network to a std::string
	void save(std::string &file, const bool readFormat = false);
	/**
	 * Loads .nn file
	 * @exception Currupt .nn files will throw an error!
	 */
	void load(std::ifstream &stream, const bool readFormat = false);
	/**
	 * Loads network from a std::string
	 * @exception Currupt .nn files will throw an error!
	 */
	void load(std::string &file, const bool readFormat = false);
	void copy(seayon *to);
	/**
	 * Combines two networks with the average values.
	 * @param with List of networks
	 * @param count How many networks
	 */
	void combine(seayon **with, size_t count);
	bool equals(seayon *second);

	// Prints all values of the network to console.
	void print();
	// Prints all values and the cost() of the network to console.
	void print(std::vector<float> &outputs);
	// Prints all values and the cost() of the network to console.
	void print(std::vector<float> &inputs, std::vector<float> &outputs);
	// Prints all values and the cost() with the accruacy() of the network to console.
	void print(std::vector<std::vector<float>> &inputs, std::vector<std::vector<float>> &outputs);
	// Prints all values and the cost() with the accruacy() of the network to console.
	void print(std::vector<std::vector<std::vector<float>>> &inputs, std::vector<std::vector<std::vector<float>>> &outputs);
	// Prints only the output layer of the network to console.
	void printo();
	// Prints the output layer and the cost() of the network to console.
	void printo(std::vector<float> &outputs);
	// Prints the output layer and the cost() of the network to console.
	void printo(std::vector<float> &inputs, std::vector<float> &outputs);
	// Prints the output layer and the cost() with the accruacy() of the network to console.
	void printo(std::vector<std::vector<float>> &inputs, std::vector<std::vector<float>> &outputs);
	// Prints the output layer and the cost() with the accruacy() of the network to console.
	void printo(std::vector<std::vector<std::vector<float>>> &inputs, std::vector<std::vector<std::vector<float>>> &outputs);

	// Calculates network outputs
	void pulse();
	/**
	 * Calculates network outputs
	 * @param inputs Set values of the input layer
	 */
	void pulse(std::vector<float> &inputs);
	/**
	 * Describes how far apart the optimal outputs from the outputs are.
	 * @param outputs This are the optimal outputs
	 * @return 0.0 = Best; 1.0 or more = Bad
	 */
	float cost(std::vector<float> &outputs);
	/**
	 * Describes how far apart the optimal outputs from the outputs are.
	 * @param inputs Set values of the input layer
	 * @param outputs This are the optimal outputs
	 * @return 0.0 = Best; 1.0 or more = Bad
	 */
	float cost(std::vector<float> &inputs, std::vector<float> &outputs);
	/**
	 * Describes how far apart the optimal outputs from the outputs are.
	 * @param inputs Set values of the input layer
	 * @param outputs This are the optimal outputs
	 * @return 0.0 = Best; 1.0 or more = Bad
	 */
	float cost(std::vector<std::vector<float>> &inputs, std::vector<std::vector<float>> &outputs);
	/**
	 * Describes how far apart the optimal outputs from the outputs are.
	 * @param inputs Set values of the input layer
	 * @param outputs This are the optimal outputs
	 * @return 0.0 = Best; 1.0 or more = Bad
	 */
	float cost(std::vector<std::vector<std::vector<float>>> &inputs, std::vector<std::vector<std::vector<float>>> &outputs);
	/**
	 * Describes how often the network would choose the right neurons to fire (neuron value >= 0.5).
	 * @param inputs Set values of the input layer
	 * @param outputs This are the optimal outputs
	 * @return 1.0 = 100%; 0.5f = 50%; 0.0 = 0%
	 */
	float accruacy(std::vector<std::vector<float>> &inputs, std::vector<std::vector<float>> &outputs);
	/**
	 * Describes how often the network would choose the right neurons to fire (neuron value >= 0.5).
	 * @param inputs Set values of the input layer
	 * @param outputs This are the optimal outputs
	 * @return 1.0 = 100%; 0.5f = 50%; 0.0 = 0%
	 */
	float accruacy(std::vector<std::vector<std::vector<float>>> &inputs, std::vector<std::vector<std::vector<float>>> &outputs);

	/**
	 * Backpropagation is an efficient training algorithm which works for many cases.
	 * @param inputs Set values of the input layer
	 * @param outputs This are the optimal outputs
	 * @param runs How many iterations. Tip: Start small with like 5 - 20
	 * @param print Writes detailed information about the progress to the console.
	 * @param learningRate Large numbers will take less runs but can "over shoot" the right value. 
	 * @param momentum Large numbers will take less runs but can "over shoot" the right value. 
	 */
	void fit(std::vector<std::vector<float>> &inputs, std::vector<std::vector<float>> &outputs, int runs, bool print = true, float learningRate = 0.03f, float momentum = 0.1f);
};