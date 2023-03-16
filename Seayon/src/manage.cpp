#include "seayon.h"

#include <string>
#include <sstream>
#include <thread>

void seayon::save(std::ofstream& file)
{
	std::string s = save();
	file << s.c_str();
	file.flush();
}
std::string seayon::save()
{
	const size_t lLast = Layers.size() - 1;
	std::stringstream ss;

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

	return ss.str();
}
void seayon::load(std::ifstream& file)
{
	const size_t lLast = Layers.size() - 1;
	file.seekg(0, file.end);
	size_t length = (size_t)file.tellg();
	file.seekg(0, file.beg);

	char* buf = new char[length];
	file.read(buf, length);

	std::stringstream ss;
	for (size_t i = 0; i < length; ++i)
		ss << buf[i];

	std::string s = ss.str();
	load(s);
}
void seayon::load(std::string s)
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
	int n1;

	while (s[p] == ' ' || s[p] == '\t' || s[p] == '\n')
		++p;

	std::stringstream sizeNumber;
	for (; s[p] != ':'; ++p)
		sizeNumber << s[p];
	++p;

	int a = std::stoul(sizeNumber.str());
	std::stringstream().swap(sizeNumber);

	Activation = (ActivFunc)a;

	for (; p < s.size(); ++p)
	{
		if (s[p] == '[')
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
					for (; s[p] != ';'; ++p)
						floatNumber << s[p];

					Layers[l2].Weights[n2][n1] = std::stof(floatNumber.str());
				}
				continue;
			}
			++depth;
		}
		else if (s[p] == '(')
		{
			++n1;

			std::stringstream floatNumber;
			++p;
			for (; s[p] != ')'; ++p)
				floatNumber << s[p];

			Layers[l1].Biases[n1] = std::stof(floatNumber.str());
		}
		else if (s[p] == ']')
			--depth;
		else
			sizeNumber << s[p];
	}
}
void seayon::copy(seayon& to)
{
	to.Layers.resize(Layers.size());
	to.Layers[0].Neurons.resize(Layers[0].Neurons.size());

	for (size_t l1 = 0; l1 < Layers.size(); ++l1)
	{
		const size_t l2 = l1 + 1;

		to.Layers[l2].Neurons.resize(Layers[l2].Neurons.size());
		to.Layers[l2].Biases.resize(Layers[l2].Neurons.size());

		if (l2 < Layers.size() - 1)
		{
			to.Layers[l2].Weights.resize(Layers[l2].Neurons.size());
			for (size_t n2 = 0; n2 < Layers[l2].Neurons.size(); ++n2)
			{
				to.Layers[l2].Weights[n2].resize(Layers[l1].Neurons.size());
				for (size_t n1 = 0; n1 < Layers[l1].Neurons.size(); ++n1)
				{
					to.Layers[l2].Weights[n2][n1] = Layers[l2].Weights[n2][n1];
				}
			}
		}
	}
}
void seayon::combine(seayon* with, size_t count)
{
	for (size_t l2 = 1; l2 < Layers.size(); ++l2)
	{
		const size_t l1 = l2 - 1;

		for (size_t n2 = 0; n2 < Layers[l2].Neurons.size(); ++n2)
		{
			float an = Layers[l2].Neurons[n2];
			for (size_t i = 0; i < count; ++i)
				an += with[i].Layers[l2].Neurons[n2];

			Layers[l2].Neurons[n2] = an / count;

			for (size_t n1 = 0; n1 < Layers[l1].Neurons.size(); ++n1)
			{
				float aw = Layers[l2].Weights[n2][n1];
				for (size_t i = 0; i < count; ++i)
					aw += with[i].Layers[l2].Weights[n2][n1];

				Layers[l2].Weights[n2][n1] = aw / count;
			}
		}
	}
	for (size_t n1 = 0; n1 < Layers[0].Neurons.size(); ++n1)
	{
		float an = Layers[0].Neurons[n1];
		for (size_t i = 0; i < count; ++i)
			an += with[i].Layers[0].Neurons[n1];

		Layers[0].Neurons[n1] = an / count;
	}
}
bool seayon::equals(seayon& second)
{
	if (Layers.size() != second.Layers.size() || Layers[0].Neurons.size() != second.Layers[0].Neurons.size())
		return 0;

	for (size_t l1 = 0; l1 < Layers.size(); ++l1)
	{
		const size_t l2 = l1 + 1;

		if (Layers[l2].Biases.size() != second.Layers[l2].Biases.size() || Layers[l2].Neurons.size() != second.Layers[l2].Neurons.size())
			return 0;

		for (size_t n2 = 0; n2 < Layers[l2].Neurons.size(); ++n2)
		{
			if (Layers[l2].Weights[n2] != second.Layers[l2].Weights[n2] || Layers[l2].Biases[n2] != second.Layers[l2].Biases[n2])
				return 0;

			if (l1 < Layers.size() - 1)
				for (size_t n1 = 0; n1 < Layers[l1].Neurons.size(); ++n1)
					if (Layers[l2].Weights[n2][n1] != second.Layers[l2].Weights[n2][n1])
						return 0;
		}
	}

	return 1;
}