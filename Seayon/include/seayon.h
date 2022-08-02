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
#define dSigmoid(a) a * (1.0f - a)

/// <summary>
/// <para>Open source Neural Network library in C++ with lots of easy to use features.</para>
/// <para>Copyright by deanqx and Sawey</para>
/// </summary>
struct seayon
{
	struct Layer
	{
		std::vector<float> Neurons;
		/// <summary>
		/// <para>Goes from first to second</para>
		/// <para>Layers[l2].Weights[n2][n1]</para>
		/// </summary>
		std::vector<std::vector<float>> Weights;
		std::vector<float> Biases;
	};
	std::vector<Layer> Layers;
	enum class ActivFunc
	{
		SIGMOID, RELU, REMOID
	} Activation = ActivFunc::SIGMOID;
	long Writing = -1;

	/// <summary>Creates simple network where every neuron is connected to each neuron in the next layer.</summary>
	/// <param name="layerCount">Starts with the input layer. Minimum 2 layers</param>
	/// <param name="func">Activation function for every neuron.</param>
	void generate(const std::vector<unsigned>& layerCounts, const ActivFunc a = ActivFunc::SIGMOID, const unsigned seed = 0);
	/// <summary>saves network to a file</summary>
	void save(std::ofstream& stream, const bool readFormat = false);
	/// <summary>Copys network to char array</summary>
	void save(std::string& stream, const bool readFormat = false);
	/// <summary>
	/// <para>Loads savefile</para>
	/// <para>WARNING: Currupt files will throw an error!</para>
	/// </summary>
	void load(std::ifstream& stream, const bool readFormat = false);
	/// <summary>
	/// <para>Loads save from string</para>
	/// <para>WARNING: Currupt files will throw an error!</para>
	/// </summary>
	void load(std::string& stream, const bool readFormat = false);
	void copy(seayon* to);
	/// <summary>
	/// <para>Averages weights and biases with anouther Network.</para>
	/// <param name="with	">If you want to average about multiple Networks.</param>
	/// </summary>
	void combine(seayon** with, size_t count);
	bool equals(seayon* second);

	/// <summary>prints values of the network to console</summary>
	void print();
	/// <summary>prints values and the cost of the network to console</summary>
	void print(std::vector<float>& outputs);
	/// <summary>prints values and the cost of the network to console</summary>
	void print(std::vector<float>& inputs, std::vector<float>& outputs);
	/// <summary>prints values and the cost of the network to console</summary>
	void print(std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& outputs);
	/// <summary>prints values and the cost of the network to console</summary>
	void print(std::vector<std::vector<std::vector<float>>>& inputs, std::vector<std::vector<std::vector<float>>>& outputs);
	/// <summary>prints outputs of the network to console</summary>
	void printo();
	/// <summary>prints outputs and the cost of the network to console</summary>
	void printo(std::vector<float>& outputs);
	/// <summary>prints outputs and the cost of the network to console</summary>
	void printo(std::vector<float>& inputs, std::vector<float>& outputs);
	/// <summary>prints outputs and the cost of the network to console</summary>
	void printo(std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& outputs);
	/// <summary>prints outputs and the cost of the network to console</summary>
	void printo(std::vector<std::vector<std::vector<float>>>& inputs, std::vector<std::vector<std::vector<float>>>& outputs);
	/// <summary>prints outputs and the cost of the network to console</summary>
	void printheat(std::ofstream& csv);

	/// <summary>Calculates network output</summary>
	void pulse();
	/// <summary>Calculates network output</summary>
	void pulse(std::vector<float>& inputs);
	/// <summary>Describes how good the network performs.</summary>
	/// <param name="outputs">Training "should" outputs</param>
	float cost(std::vector<float>& outputs);
	/// <summary>Describes how good the network performs.</summary>
	/// <param name="inputs">Training inputs, every training in/output needs a fellow.</param>
	/// <param name="outputs">Training "should" outputs, every training in/output needs a fellow.</param>
	float cost(std::vector<float>& inputs, std::vector<float>& outputs);
	/// <summary>Describes how good the network performs.</summary>
	/// <param name="inputs">Training inputs, every training in/output needs a fellow.</param>
	/// <param name="outputs">Training "should" outputs, every training in/output needs a fellow.</param>
	float cost(std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& outputs);
	/// <summary>Describes how good the network performs.</summary>
	/// <param name="inputs">Training inputs, every training in/output needs a fellow.</param>
	/// <param name="outputs">Training "should" outputs, every training in/output needs a fellow.</param>
	float cost(std::vector<std::vector<std::vector<float>>>& inputs, std::vector<std::vector<std::vector<float>>>& outputs);
	float accruacy(std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& outputs);
	float accruacy(std::vector<std::vector<std::vector<float>>>& inputs, std::vector<std::vector<std::vector<float>>>& outputs);

	/// <summary>Backpropagation is a very efficient training algorithm.</summary>
	/// <param name="inputs">Training inputs, every training in/output needs a fellow.</param>
	/// <param name="outputs">Training should output, every training in/output needs a fellow.</param>
	/// <param name="runs">Start with less</param>
	/// <param name="writelog">Writes detailed information about the progress.</param>
	/// <param name="learningRate">Too high numbers can over shoot the right weight.</param>
	/// <param name="momentum">How much the algorithm is allowed to move.</param>
	void fit(std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& outputs, unsigned runs, bool print = true, float learningRate = 0.03f, float momentum = 0.1f);
};