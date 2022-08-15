#include "seayon.h"

#include <string>
#include <sstream>
#include <thread>

#define randf(MIN, MAX) MIN + (float)rand() / (float)(RAND_MAX / (MAX - MIN))
void seayon::generate(const std::vector<int>& layerCounts, const ActivFunc a, const int seed)
{
	Activation = a;

	if (seed != 0)
		srand(seed);

	std::vector<Layer> flashVector;
	Layers.swap(flashVector);

	Layers.resize(layerCounts.size());
	Layers[0].Neurons.resize(layerCounts[0]);

	for (size_t l2 = 1; l2 < layerCounts.size(); ++l2)
	{
		const size_t l1 = l2 - 1;

		Layers[l2].Neurons.resize(layerCounts[l2]);
		Layers[l2].Biases.resize(layerCounts[l2]);

		if (l2 < layerCounts.size())
		{
			Layers[l2].Weights.resize(layerCounts[l2]);
			for (size_t n2 = 0; n2 < layerCounts[l2]; ++n2)
			{
				Layers[l2].Weights[n2].resize(layerCounts[l1]);
				for (size_t n1 = 0; n1 < layerCounts[l1]; ++n1)
				{
					Layers[l2].Weights[n2][n1] = randf(-2.0f, 2.0f);
				}
			}
		}
	}
}
void seayon::save(std::ofstream& stream, const bool readFormat)
{
	std::string file;
	save(file, readFormat);

	stream << file.c_str();
	stream.flush();
}
void seayon::save(std::string& stream, const bool readFormat)
{
	const size_t lLast = Layers.size() - 1;
	std::stringstream ss;

	if (readFormat)
	{
		ss << (int)Activation << ":\n" << Layers.size() << " [\n";
		for (size_t l2 = 1; l2 < Layers.size(); ++l2)
		{
			size_t l1 = l2 - 1;

			ss << '\t' << Layers[l1].Neurons.size() << " [\n";
			for (size_t n1 = 0; n1 < Layers[l1].Neurons.size(); ++n1)
			{
				if (l1 == 0)
					ss << "\t\t" << Layers[l2].Neurons.size() << " [\n";
				else
					ss << "\t\t" << Layers[l2].Neurons.size() << "(" << Layers[l1].Biases[n1] << ") [\n";
				for (size_t n2 = 0; n2 < Layers[l2].Neurons.size(); ++n2)
				{
					ss << "\t\t\t" << Layers[l2].Weights[n2][n1] << ";\n";
				}
				ss << "\t\t]\n";
			}
			ss << "\t]\n";
		}

		ss << '\t' << Layers[lLast].Neurons.size() << " [\n";
		for (size_t n2 = 0; n2 < Layers[lLast].Neurons.size(); ++n2)
		{
			ss << "\t\t(" << Layers[lLast].Biases[n2] << ")\n";
		}
		ss << "\t]\n]";
	}
	else
	{

		ss << (int)Activation << ":" << Layers.size() << '[';
		for (size_t l2 = 1; l2 < Layers.size(); ++l2)
		{
			size_t l1 = l2 - 1;

			ss << Layers[l1].Neurons.size() << '[';
			for (size_t n1 = 0; n1 < Layers[l1].Neurons.size(); ++n1)
			{
				if (l1 == 0)
					ss << Layers[l2].Neurons.size() << '[';
				else
					ss << Layers[l2].Neurons.size() << '(' << Layers[l1].Biases[n1] << ")[";
				for (size_t n2 = 0; n2 < Layers[l2].Neurons.size(); ++n2)
				{
					ss << Layers[l2].Weights[n2][n1] << ';';
				}
				ss << "]";
			}
			ss << "]";
		}

		ss << Layers[lLast].Neurons.size() << '[';
		for (size_t n2 = 0; n2 < Layers[lLast].Neurons.size(); ++n2)
		{
			ss << '(' << Layers[lLast].Biases[n2] << ")";
		}
		ss << "]]";
	}

	stream = ss.str();
}
void seayon::load(std::ifstream& stream, const bool readFormat)
{
	const size_t lLast = Layers.size() - 1;
	stream.seekg(0, stream.end);
	size_t length = (size_t)stream.tellg();
	stream.seekg(0, stream.beg);

	char* buf = new char[length];
	stream.read(buf, length);

	std::stringstream ss;
	for (size_t i = 0; i < length; ++i)
		ss << buf[i];

	std::string s = ss.str();
	load(s, readFormat);
}
void seayon::load(std::string& stream, const bool readFormat)
{
	std::vector<Layer> flashVector;
	Layers.swap(flashVector);

	/// <summary>
	/// <para>0: Layout</para>
	/// <para>1: Layer</para>
	/// <para>2: Neuron</para>
	/// </summary>
	size_t p = 0;
	int depth = 0;
	int l1 = -1;
	int n1{};

	while (stream[p] == ' ' || stream[p] == '\t' || stream[p] == '\n')
		++p;

	std::stringstream sizeNumber;
	for (; stream[p] != ':'; ++p)
		sizeNumber << stream[p];
	++p;

	int a = std::stoul(sizeNumber.str());
	std::stringstream().swap(sizeNumber);

	Activation = (ActivFunc)a;

	if (readFormat)
	{
		for (; p < stream.size(); ++p)
		{
			while (stream[p] == ' ' || stream[p] == '\t' || stream[p] == '\n')
				++p;

			if (stream[p] == '[')
			{
				if (depth == 0)
				{
					int layerCount = std::stoul(sizeNumber.str());
					std::stringstream().swap(sizeNumber);

					Layers.resize(layerCount);
				}
				else if (depth == 1)
				{
					int neuronCount = std::stoul(sizeNumber.str());
					std::stringstream().swap(sizeNumber);

					++l1;
					n1 = -1;

					Layers[l1].Neurons.resize(neuronCount);
					if (l1 != 0)
						Layers[l1].Biases.resize(neuronCount);
				}
				else if (depth == 2)
				{
					if (l1 == 0)
						++n1;
					++depth;

					int weightCount = std::stoul(sizeNumber.str());
					std::stringstream().swap(sizeNumber);

					int l2 = l1 + 1;

					Layers[l2].Weights.resize(weightCount);
					for (size_t n2 = 0; n2 < weightCount; ++n2)
					{
						if (Layers[l2].Weights[n2].size() < Layers[l1].Neurons.size()) // PERF Not a good solution
							Layers[l2].Weights[n2].resize(Layers[l1].Neurons.size());

						std::stringstream floatNumber;
						++p;
						for (; stream[p] != ';'; ++p)
						{
							while (stream[p] == ' ' || stream[p] == '\t' || stream[p] == '\n')
								++p;

							floatNumber << stream[p];
						}

						Layers[l2].Weights[n2][n1] = std::stof(floatNumber.str());
					}
					continue;
				}
				++depth;
			}
			else if (stream[p] == '(')
			{
				++n1;

				std::stringstream floatNumber;
				++p;
				for (; stream[p] != ')'; ++p)
				{
					while (stream[p] == ' ' || stream[p] == '\t' || stream[p] == '\n')
						++p;

					floatNumber << stream[p];
				}

				Layers[l1].Biases[n1] = std::stof(floatNumber.str());
			}
			else if (stream[p] == ']')
				--depth;
			else
				sizeNumber << stream[p];
		}
	}
	else
	{
		for (; p < stream.size(); ++p)
		{
			if (stream[p] == '[')
			{
				if (depth == 0)
				{
					int layerCount = std::stoul(sizeNumber.str());
					std::stringstream().swap(sizeNumber);

					Layers.resize(layerCount);
				}
				else if (depth == 1)
				{
					int neuronCount = std::stoul(sizeNumber.str());
					std::stringstream().swap(sizeNumber);

					++l1;
					n1 = -1;

					Layers[l1].Neurons.resize(neuronCount);
					if (l1 != 0)
						Layers[l1].Biases.resize(neuronCount);
				}
				else if (depth == 2)
				{
					if (l1 == 0)
						++n1;
					++depth;

					int weightCount = std::stoul(sizeNumber.str());
					std::stringstream().swap(sizeNumber);

					int l2 = l1 + 1;

					Layers[l2].Weights.resize(weightCount);
					for (size_t n2 = 0; n2 < weightCount; ++n2)
					{
						if (Layers[l2].Weights[n2].size() < Layers[l1].Neurons.size()) // PERF Not a good solution
							Layers[l2].Weights[n2].resize(Layers[l1].Neurons.size());

						std::stringstream floatNumber;
						++p;
						for (; stream[p] != ';'; ++p)
							floatNumber << stream[p];

						Layers[l2].Weights[n2][n1] = std::stof(floatNumber.str());
					}
					continue;
				}
				++depth;
			}
			else if (stream[p] == '(')
			{
				++n1;

				std::stringstream floatNumber;
				++p;
				for (; stream[p] != ')'; ++p)
					floatNumber << stream[p];

				Layers[l1].Biases[n1] = std::stof(floatNumber.str());
			}
			else if (stream[p] == ']')
				--depth;
			else
				sizeNumber << stream[p];
		}
	}
}
void seayon::copy(seayon* to)
{
	long threadID = (long)std::hash<std::thread::id>{}(std::this_thread::get_id()); // WARN Not save to hash
	if (to->Writing < -1)
	{
	Retry:
		while (to->Writing == -1);

		to->Writing = threadID;
		if (to->Writing != threadID)
			goto Retry;
	}

	to->Layers.resize(Layers.size());
	to->Layers[0].Neurons.resize(Layers[0].Neurons.size());

	for (size_t l1 = 0; l1 < Layers.size(); ++l1)
	{
		const size_t l2 = l1 + 1;

		to->Layers[l2].Neurons.resize(Layers[l2].Neurons.size());
		to->Layers[l2].Biases.resize(Layers[l2].Neurons.size());

		if (l2 < Layers.size() - 1)
		{
			to->Layers[l2].Weights.resize(Layers[l2].Neurons.size());
			for (size_t n2 = 0; n2 < Layers[l2].Neurons.size(); ++n2)
			{
				to->Layers[l2].Weights[n2].resize(Layers[l1].Neurons.size());
				for (size_t n1 = 0; n1 < Layers[l1].Neurons.size(); ++n1)
				{
					to->Layers[l2].Weights[n2][n1] = Layers[l2].Weights[n2][n1];
				}
			}
		}
	}

	to->Writing = -1;
}
void seayon::combine(seayon** with, size_t count)
{
	long threadID = (long)std::hash<std::thread::id>{}(std::this_thread::get_id()); // WARN Not save to hash
	if (Writing < -1)
	{
	Retry:
		while (Writing == -1);

		Writing = threadID;
		if (Writing != threadID)
			goto Retry;
	}

	for (size_t l2 = 1; l2 < Layers.size(); ++l2)
	{
		const size_t l1 = l2 - 1;

		for (size_t n2 = 0; n2 < Layers[l2].Neurons.size(); ++n2)
		{
			float an = Layers[l2].Neurons[n2];
			for (size_t i = 0; i < count; ++i)
				an += with[i]->Layers[l2].Neurons[n2];

			Layers[l2].Neurons[n2] = an / count;

			for (size_t n1 = 0; n1 < Layers[l1].Neurons.size(); ++n1)
			{
				float aw = Layers[l2].Weights[n2][n1];
				for (size_t i = 0; i < count; ++i)
					aw += with[i]->Layers[l2].Weights[n2][n1];

				Layers[l2].Weights[n2][n1] = aw / count;
			}
		}
	}
	for (size_t n1 = 0; n1 < Layers[0].Neurons.size(); ++n1)
	{
		float an = Layers[0].Neurons[n1];
		for (size_t i = 0; i < count; ++i)
			an += with[i]->Layers[0].Neurons[n1];

		Layers[0].Neurons[n1] = an / count;
	}

	Writing = -1;
}
bool seayon::equals(seayon* second)
{
	if (Layers.size() != second->Layers.size() || Layers[0].Neurons.size() != second->Layers[0].Neurons.size())
		return 0;

	for (size_t l1 = 0; l1 < Layers.size(); ++l1)
	{
		const size_t l2 = l1 + 1;

		if (Layers[l2].Biases.size() != second->Layers[l2].Biases.size() || Layers[l2].Neurons.size() != second->Layers[l2].Neurons.size())
			return 0;

		for (size_t n2 = 0; n2 < Layers[l2].Neurons.size(); ++n2)
		{
			if (Layers[l2].Weights[n2] != second->Layers[l2].Weights[n2] || Layers[l2].Biases[n2] != second->Layers[l2].Biases[n2])
				return 0;

			if (l1 < Layers.size() - 1)
				for (size_t n1 = 0; n1 < Layers[l1].Neurons.size(); ++n1)
					if (Layers[l2].Weights[n2][n1] != second->Layers[l2].Weights[n2][n1])
						return 0;
		}
	}

	return 1;
}