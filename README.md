![Console Screenshot](https://raw.githubusercontent.com/deanqx/Seayon/stable/preview.png)

# Version 1.1

:heavy_check_mark: New optimizer: Mini batches

:heavy_check_mark: GPU accelerated training

:heavy_check_mark: Improved data structure

:heavy_check_mark: Different activation function for every layer

:heavy_check_mark: Fast saving and loading

# Getting started

Project is built using Visual Studio Code on Windows.

The default branch contains the newest stable version. The currently in development beta version is also available.

```
git clone https://github.com/deanqx/Seayon

git submodule init

git submodule update
```

**Optional:** To use the beta version you can switch like this:

```
git switch beta
```

## **Quick start**

Example project: `SeayonDemo`

## **Including in your Project**

To use Seayon in your project you include the basis header:

```C++
#include "seayon.hpp"
```

To use the GPU accelerated version you include the cuda header and install the [Cuda toolkit](https://developer.nvidia.com/cuda-downloads). Follow the [Cuda Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows) to setup the compiler.

```C++
#include "cuda_seayon.cuh"
```

**Note:** When you are using `SeayonMnist` you have to download the `Mnist dataset` and put it into the `SeayonMnist/res` folder.

# Found a bug

If you have found an issue or you would like to submit improvements, use the **Issues tab**, select the version you are using under "milestones" and choose fitting labels.
The **Issues tab** can also be used to start discussions. Make sure beforehand whether the issue has already been fixed in the beta version.

# Contributing

I am very grateful for any contributions. To start fork the repository to create your own copy, when you're done you can create a pull request. You can only contribute to the latest beta version and not directly into the stable branch.

Before opening a pull request, please submit an issue explaining the bug or feature and reference the [issue hashtag](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/autolinked-references-and-urls#issues-and-pull-requests) in the discription.
