# Seayon

Open source Convolutional Neural Network library in C++ with lots of easy to use features.
The **main branch** contains the newest stable version.
You can find the work-in-progress version in the **beta branch**.

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

:x: Only **Sigmoid** or **ReLu** as activation functions.

:x: No

# TODO-List

- Save and load in Json format
- Continue SeayonTrading
- Add multi threading support
- Add more activation functions

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

```C++
#include <vector>
#include "seayon.h"

int main()
{
	seayon::trainingdata data = {std::vector<seayon::trainingdata::sample>{
		{std::vector<float>{1.0f, 0.0f}, std::vector<float>{0.0f, 1.0f}},
		{std::vector<float>{0.0f, 1.0f}, std::vector<float>{1.0f, 0.0f}}}}; // Samples[x]: {inputs, outputs}

	seayon *nn = new seayon; // nn = neural network

	// Input layer size: 2
	// 1. Hidden layer size: 3
	// 2. Hidden layer size: 4
	// Output layer size: 2
	nn->generate(std::vector<int>{2, 3, 4, 2}, seayon::ActivFunc::SIGMOID, 1472); // Randomization seed: 1472

	nn->print(data, 0); // Prints the whole network to the console

	// ### Before training ###
	nn->printo(data, 0); // Prints only the output layer to the console

	// 50 iterations | 0.5f learning rate | 0.5f momentum
	nn->fit(data, 50, false, nullptr, 0.5f, 0.5f);

	// ### After training ###
	nn->printo(data, 0);

	return 0;
}
```

Screenshot of the console with learning mnist in progress:

![Console Screenshot](https://raw.githubusercontent.com/deanqx/Seayon/main/image.png)

# Found a bug

If you found an issue or you would like to submit improvements, use the **Issues tab**.
You can reach me at dean@kowatsch.de

# About me

My name is Dean Schneider from Germany. I were 14 years old when I started this project. Since then this was my main and favorite project of all time. In Juli 2022 I switched from Visual Studio 2019 to Visual Studio Code and started using github.

I started programming in 2015 with an age of 9. At this time I bearly had any english lessons in school. I was watching YouTube on my laptop when YouTube recommeded me a video about reading out the Wi-Fi password with the cmd console. That sounded interesting. In the following months I learned slow but steady about Windows batch. After 1 and a half year I finished my first batch/cmd project called pSon. I recorded a pretty awkward YouTube video about the finished applicion (<https://youtu.be/ECsoVTyEcIQ>). I continued with learning Java then C++ and many other languages. I have made many small projects mostly server/client applications or small Unity experiments. In November 2020 I created the project Seayon Ai.
