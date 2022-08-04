# Seayon
Open source Convolutional Neural Network library in C++ with lots of easy to use features.
The **master branch** contains the newest stable version.
You can find the work-in-progress version in the **test branch**.

# Features
:heavy_check_mark: **Mostly performance optimized**

:heavy_check_mark: Generate a network with a **custom amount of layers and neurons**.

:heavy_check_mark: **Save and load** networks from a Json or Seayon's compact and fast file format.

:heavy_check_mark: **Compare** networks with eachother.

:heavy_check_mark: **Print** a whole network to the console.

:heavy_check_mark: **Print** only the output layer to console.

:heavy_check_mark: Calculate the **cost** or the **accruacy** of a network.

:heavy_check_mark: Train with the **Backpropagation** algorithm.

:x: No **GPU** or **Multi-threading** support

:x: Uses `std::vector`

:x: Only **Sigmoid** and **ReLu** as activation functions.

# TODO-List
1. Save and load in Json format
- Continue SeayonTrading
- Replace vectors with arrays
- Implement remoid

# How to include the library
1. Copy `./bin/Seayon/include/` and `./bin/Seayon/lib/` to your workspace folder.
2. `#include "seayon.h"`
3. Use `seayon* nn = new seayon;` to allocate on the heap, which is slower but prevents memory overflows.
4. If you're using g++ add following `tags`:
   - `-l Seayon` **Seayon** actually means **libSeayon**
   - `-I ./include`
   - `-L ./lib`
   - `-static` **(Optional)**
   - **Example:** `g++ main.cpp -o main -l Seayon -I ./include -L ./lib -static`

# Be aware of
The **Sigmoid** activation function is **strongly recommended**: **ReLu** can result in memory overflows.

The current code is made for **Windows**.

### :warning: Seayon is an open source project and can used in your applications, but
- **please don't rename Seayon when using in other projects.**
- **please keep the copyright in the source code.**

# How to start
### 1. Demo program
You can find the following code in `./SeayonDemo/main.cpp`. Everything is explained in the code.

````C++
#include <vector>
#include "seayon.h"

int main()
{
	std::vector<std::vector<float>> inputs = {
		{1.0f, 0.0f},
		{0.0f, 1.0f}}; // Two sets of "Input layer values"
	std::vector<std::vector<float>> outputs = {
		{0.0f, 1.0f},
		{1.0f, 0.0f}}; // Two sets of "Input layer values"

	seayon *nn = new seayon; // nn = neural network

	// Input layer size: 2
	// 1. Hidden layer size: 3
	// 2. Hidden layer size: 4
	// Output layer size: 2
	nn->generate(std::vector<int>{2, 3, 4, 2}, seayon::ActivFunc::SIGMOID);

	nn->pulse(inputs[0]); // Calculates the network with first input set
	nn->print();  // Prints the whole network to the console

	// ### Before training ###
	nn->pulse(inputs[0]);
	nn->printo();
	nn->pulse(inputs[1]);
	nn->printo(inputs, outputs); // Prints only the output layer to the console


	// 100 iterations | 0.5f learning rate | 0.5f momentum
	nn->fit(inputs, outputs, 100, 0.5f, 0.5f);


	// ### After training ###
	nn->pulse(inputs[0]);
	nn->printo();
	nn->pulse(inputs[1]);
	nn->printo(inputs, outputs);

	return 0;
}
````
Screenshot of the console with training in progress:

![Console Screenshot](https://github.com/deanqx/Seayon/blob/master/image.png?raw=true)

### 2. Recognise handwritten digits
You can find a second example project in `./SeayonMnist` this one is able to learn how to recognise handwritten digits.

#### :warning: Recommended for Experienced Ai Programmers

1. Download the **MNIST handwritten digit dataset** (mnist_training.csv 107.0 KB, mnist_test.csv 17.8 KB) or any other handwritten digit dataset
2. Create following directories: `./SeayonMnist/res` and `./SeayonMnist/res/mnist`
3. Place `mnist_training.csv` and `mnist_test.csv` in your `./SeayonMnist/res/mnist` folder.
4. Now insert this at the begin of `mnist_training.csv` and `mnist_test.csv`:
````c
number,0;0,0;1,0;2,0;3,0;4,0;5,0;6,0;7,0;8,0;9,0;10,0;11,0;12,0;13,0;14,0;15,0;16,0;17,0;18,0;19,0;20,0;21,0;22,0;23,0;24,0;25,0;26,0;27,1;0,1;1,1;2,1;3,1;4,1;5,1;6,1;7,1;8,1;9,1;10,1;11,1;12,1;13,1;14,1;15,1;16,1;17,1;18,1;19,1;20,1;21,1;22,1;23,1;24,1;25,1;26,1;27,2;0,2;1,2;2,2;3,2;4,2;5,2;6,2;7,2;8,2;9,2;10,2;11,2;12,2;13,2;14,2;15,2;16,2;17,2;18,2;19,2;20,2;21,2;22,2;23,2;24,2;25,2;26,2;27,3;0,3;1,3;2,3;3,3;4,3;5,3;6,3;7,3;8,3;9,3;10,3;11,3;12,3;13,3;14,3;15,3;16,3;17,3;18,3;19,3;20,3;21,3;22,3;23,3;24,3;25,3;26,3;27,4;0,4;1,4;2,4;3,4;4,4;5,4;6,4;7,4;8,4;9,4;10,4;11,4;12,4;13,4;14,4;15,4;16,4;17,4;18,4;19,4;20,4;21,4;22,4;23,4;24,4;25,4;26,4;27,5;0,5;1,5;2,5;3,5;4,5;5,5;6,5;7,5;8,5;9,5;10,5;11,5;12,5;13,5;14,5;15,5;16,5;17,5;18,5;19,5;20,5;21,5;22,5;23,5;24,5;25,5;26,5;27,6;0,6;1,6;2,6;3,6;4,6;5,6;6,6;7,6;8,6;9,6;10,6;11,6;12,6;13,6;14,6;15,6;16,6;17,6;18,6;19,6;20,6;21,6;22,6;23,6;24,6;25,6;26,6;27,7;0,7;1,7;2,7;3,7;4,7;5,7;6,7;7,7;8,7;9,7;10,7;11,7;12,7;13,7;14,7;15,7;16,7;17,7;18,7;19,7;20,7;21,7;22,7;23,7;24,7;25,7;26,7;27,8;0,8;1,8;2,8;3,8;4,8;5,8;6,8;7,8;8,8;9,8;10,8;11,8;12,8;13,8;14,8;15,8;16,8;17,8;18,8;19,8;20,8;21,8;22,8;23,8;24,8;25,8;26,8;27,9;0,9;1,9;2,9;3,9;4,9;5,9;6,9;7,9;8,9;9,9;10,9;11,9;12,9;13,9;14,9;15,9;16,9;17,9;18,9;19,9;20,9;21,9;22,9;23,9;24,9;25,9;26,9;27,10;0,10;1,10;2,10;3,10;4,10;5,10;6,10;7,10;8,10;9,10;10,10;11,10;12,10;13,10;14,10;15,10;16,10;17,10;18,10;19,10;20,10;21,10;22,10;23,10;24,10;25,10;26,10;27,11;0,11;1,11;2,11;3,11;4,11;5,11;6,11;7,11;8,11;9,11;10,11;11,11;12,11;13,11;14,11;15,11;16,11;17,11;18,11;19,11;20,11;21,11;22,11;23,11;24,11;25,11;26,11;27,12;0,12;1,12;2,12;3,12;4,12;5,12;6,12;7,12;8,12;9,12;10,12;11,12;12,12;13,12;14,12;15,12;16,12;17,12;18,12;19,12;20,12;21,12;22,12;23,12;24,12;25,12;26,12;27,13;0,13;1,13;2,13;3,13;4,13;5,13;6,13;7,13;8,13;9,13;10,13;11,13;12,13;13,13;14,13;15,13;16,13;17,13;18,13;19,13;20,13;21,13;22,13;23,13;24,13;25,13;26,13;27,14;0,14;1,14;2,14;3,14;4,14;5,14;6,14;7,14;8,14;9,14;10,14;11,14;12,14;13,14;14,14;15,14;16,14;17,14;18,14;19,14;20,14;21,14;22,14;23,14;24,14;25,14;26,14;27,15;0,15;1,15;2,15;3,15;4,15;5,15;6,15;7,15;8,15;9,15;10,15;11,15;12,15;13,15;14,15;15,15;16,15;17,15;18,15;19,15;20,15;21,15;22,15;23,15;24,15;25,15;26,15;27,16;0,16;1,16;2,16;3,16;4,16;5,16;6,16;7,16;8,16;9,16;10,16;11,16;12,16;13,16;14,16;15,16;16,16;17,16;18,16;19,16;20,16;21,16;22,16;23,16;24,16;25,16;26,16;27,17;0,17;1,17;2,17;3,17;4,17;5,17;6,17;7,17;8,17;9,17;10,17;11,17;12,17;13,17;14,17;15,17;16,17;17,17;18,17;19,17;20,17;21,17;22,17;23,17;24,17;25,17;26,17;27,18;0,18;1,18;2,18;3,18;4,18;5,18;6,18;7,18;8,18;9,18;10,18;11,18;12,18;13,18;14,18;15,18;16,18;17,18;18,18;19,18;20,18;21,18;22,18;23,18;24,18;25,18;26,18;27,19;0,19;1,19;2,19;3,19;4,19;5,19;6,19;7,19;8,19;9,19;10,19;11,19;12,19;13,19;14,19;15,19;16,19;17,19;18,19;19,19;20,19;21,19;22,19;23,19;24,19;25,19;26,19;27,20;0,20;1,20;2,20;3,20;4,20;5,20;6,20;7,20;8,20;9,20;10,20;11,20;12,20;13,20;14,20;15,20;16,20;17,20;18,20;19,20;20,20;21,20;22,20;23,20;24,20;25,20;26,20;27,21;0,21;1,21;2,21;3,21;4,21;5,21;6,21;7,21;8,21;9,21;10,21;11,21;12,21;13,21;14,21;15,21;16,21;17,21;18,21;19,21;20,21;21,21;22,21;23,21;24,21;25,21;26,21;27,22;0,22;1,22;2,22;3,22;4,22;5,22;6,22;7,22;8,22;9,22;10,22;11,22;12,22;13,22;14,22;15,22;16,22;17,22;18,22;19,22;20,22;21,22;22,22;23,22;24,22;25,22;26,22;27,23;0,23;1,23;2,23;3,23;4,23;5,23;6,23;7,23;8,23;9,23;10,23;11,23;12,23;13,23;14,23;15,23;16,23;17,23;18,23;19,23;20,23;21,23;22,23;23,23;24,23;25,23;26,23;27,24;0,24;1,24;2,24;3,24;4,24;5,24;6,24;7,24;8,24;9,24;10,24;11,24;12,24;13,24;14,24;15,24;16,24;17,24;18,24;19,24;20,24;21,24;22,24;23,24;24,24;25,24;26,24;27,25;0,25;1,25;2,25;3,25;4,25;5,25;6,25;7,25;8,25;9,25;10,25;11,25;12,25;13,25;14,25;15,25;16,25;17,25;18,25;19,25;20,25;21,25;22,25;23,25;24,25;25,25;26,25;27,26;0,26;1,26;2,26;3,26;4,26;5,26;6,26;7,26;8,26;9,26;10,26;11,26;12,26;13,26;14,26;15,26;16,26;17,26;18,26;19,26;20,26;21,26;22,26;23,26;24,26;25,26;26,26;27,27;0,27;1,27;2,27;3,27;4,27;5,27;6,27;7,27;8,27;9,27;10,27;11,27;12,27;13,27;14,27;15,27;16,27;17,27;18,27;19,27;20,27;21,27;22,27;23,27;24,27;25,27;26,27;27
````
5. Change `std::string workspaceFolder = "..."` to your workspace folder.
6. Make sure that `load = false`
7. I compile with g++ **BE SURE** to use the right paths.
```
g++ -O3 ./Seayon/SeayonDemo/src/main.cpp -o ./Seayon/bin/SeayonMnist/SeayonMnist -l Seayon -I ./Seayon/bin/Seayon/include -L ./Seayon/bin/Seayon/lib -static
```

# Found a bug
If you found an issue or you would like to submit improvements, use the **Issues tab**.
You can reach me at dean@kowatsch.de

# About me
My name is Dean Schneider from Germany. I were 14 years old when I started this project. Since then this was my main and favorite project of all time. In Juli 2022 I switched from Visual Studio 2019 to Visual Studio Code and started using github.

I started programming in 2015 with an age of 9. At this time I bearly had any english lessons in school. I was watching YouTube on my laptop when YouTube recommeded me a video about reading out the Wi-Fi password with the cmd console. That sounded interesting. In the following months I learned slow but steady about Windows batch. After 1 and a half year I finished my first batch/cmd project called pSon. I recorded a pretty awkward YouTube video about the finished applicion (<https://youtu.be/ECsoVTyEcIQ>). I continued with learning Java then C++ and many other languages. I have made many small projects mostly server/client applications or small Unity experiments. In November 2020 I created the project Seayon Ai.
