#include "seayon.h"

void seayon::print()
{
	HANDLE cmd = GetStdHandle(STD_OUTPUT_HANDLE);

	int normalColor;

	for (size_t l1 = 0; l1 < Layers.size(); ++l1)
	{
		const size_t l2 = l1 + 1;

		if (l1 == 0)
		{
			normalColor = 7;
			SetConsoleTextAttribute(cmd, 7);

			printf("\n  Input Layer:\n");
		}
		else if (l1 == Layers.size() - 1)
		{
			normalColor = 11;
			SetConsoleTextAttribute(cmd, 11);

			printf("  Output Layer:\n");
		}
		else
		{
			normalColor = 8;
			SetConsoleTextAttribute(cmd, 8);

			printf("  Hidden Layer[%zu]:\n", l1 - 1);
		}

		for (size_t n1 = 0; n1 < Layers[l1].Neurons.size(); ++n1)
		{
			printf("\t\tNeuron[%02zu]   ", n1);
			if (l1 == Layers.size() - 1)
			{
				if (Layers[l1].Neurons[n1] > 0.50f)
					SetConsoleTextAttribute(cmd, 95);
				else
					SetConsoleTextAttribute(cmd, 7);
				printf("%.2f", Layers[l1].Neurons[n1]);
				SetConsoleTextAttribute(cmd, normalColor);
			}
			else
				printf("%0.2f", Layers[l1].Neurons[n1]);

			if (l1 > 0)
			{
				if (Layers[l1].Biases[n1] <= 0.0f)
				{
					printf("\t\t(");
					SetConsoleTextAttribute(cmd, 12);
					printf("%0.2f", Layers[l1].Biases[n1]);
					SetConsoleTextAttribute(cmd, normalColor);
					printf(")\n");
				}
				else
					printf("\t\t(%0.2f)\n", Layers[l1].Biases[n1]);
			}
			else
				printf("\n");

			if (l1 < Layers.size() - 1)
				for (size_t n2 = 0; n2 < Layers[l2].Neurons.size(); ++n2)
				{
					printf("\t\t  Weight[%02zu] ", n2);
					if (Layers[l2].Weights[n2][n1] <= 0.0f)
					{
						SetConsoleTextAttribute(cmd, 12);
						printf("%.2f\n", Layers[l2].Weights[n2][n1]);
						SetConsoleTextAttribute(cmd, normalColor);
					}
					else
						printf("%.2f\n", Layers[l2].Weights[n2][n1]);
				}
			printf("\n");
		}
	}
	SetConsoleTextAttribute(cmd, 7);
	printf("\t-----------------------------------------------\n\n");
}
void seayon::print(std::vector<float>& outputs)
{
	print();
	printf("\t\tCost\t\t%.3f\n", cost(outputs));
	printf("\t-----------------------------------------------\n\n");
}
void seayon::print(std::vector<float>& inputs, std::vector<float>& outputs)
{
	print();
	printf("\t\tCost\t\t%.3f\n", cost(inputs, outputs));
	printf("\t-----------------------------------------------\n\n");
}
void seayon::print(std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& outputs)
{
	print();
	printf("\t\tCost\t\t%.3f\n", cost(inputs, outputs));
	printf("\t\tAccruacy\t%.1f%%\n", accruacy(inputs, outputs) * 100.0f);
	printf("\t-----------------------------------------------\n\n");
}
void seayon::print(std::vector<std::vector<std::vector<float>>>& inputs, std::vector<std::vector<std::vector<float>>>& outputs)
{
	print();
	printf("\t\tCost\t\t%.3f\n", cost(inputs, outputs));
	printf("\t\tAccruacy\t%.1f%%\n", accruacy(inputs, outputs) * 100.0f);
	printf("\t-----------------------------------------------\n\n");
}

void seayon::printo()
{
	HANDLE cmd = GetStdHandle(STD_OUTPUT_HANDLE);

	int normalColor = 11;

	size_t l = Layers.size() - 1;

	SetConsoleTextAttribute(cmd, 11);
	printf("\t  Output Layer:\n");

	for (size_t n = 0; n < Layers[l].Neurons.size(); ++n)
	{
		printf("\t\tNeuron[%.2zu]   ", n);
		if (l == Layers.size() - 1)
		{
			if (Layers[l].Neurons[n] > 0.50f)
				SetConsoleTextAttribute(cmd, 95);
			else
				SetConsoleTextAttribute(cmd, 7);
			printf("%.2f", Layers[l].Neurons[n]);
			SetConsoleTextAttribute(cmd, normalColor);
		}
		else
			printf("%.2f", Layers[l].Neurons[n]);

		if (l > 0)
		{
			if (Layers[l].Biases[n] <= 0.0f)
			{
				printf("\t\t(");
				SetConsoleTextAttribute(cmd, 12);
				printf("%0.2f", Layers[l].Biases[n]);
				SetConsoleTextAttribute(cmd, normalColor);
				printf(")\n");
			}
			else
				printf("\t\t(%0.2f)\n", Layers[l].Biases[n]);
		}
		else
			printf("\n\n");
	}
	SetConsoleTextAttribute(cmd, 7);
	printf("\t-----------------------------------------------\n\n");
}
void seayon::printo(std::vector<float>& outputs)
{
	printo();
	printf("\t\tCost\t\t%.3f\n", cost(outputs));
	printf("\t-----------------------------------------------\n\n");
}
void seayon::printo(std::vector<float>& inputs, std::vector<float>& outputs)
{
	printo();
	printf("\t\tCost\t\t%.3f\n", cost(inputs, outputs));
	printf("\t-----------------------------------------------\n\n");
}
void seayon::printo(std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& outputs)
{
	printo();
	printf("\t\tCost\t\t%.3f\n", cost(inputs, outputs));
	printf("\t\tAccruacy\t%.1f%%\n", accruacy(inputs, outputs) * 100.0f);
	printf("\t-----------------------------------------------\n\n");
}
void seayon::printo(std::vector<std::vector<std::vector<float>>>& inputs, std::vector<std::vector<std::vector<float>>>& outputs)
{
	printo();
	printf("\t\tCost\t\t%.3f\n", cost(inputs, outputs));
	printf("\t\tAccruacy\t%.1f%%\n", accruacy(inputs, outputs) * 100.0f);
	printf("\t-----------------------------------------------\n\n");
}
void seayon::printheat(std::ofstream& csv)
{
}