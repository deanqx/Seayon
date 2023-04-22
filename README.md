![Console Screenshot](https://raw.githubusercontent.com/deanqx/Seayon/stable/preview.png)

# Version 1.1

:heavy_check_mark: New optimizer: Mini batches

:heavy_check_mark: GPU accelerated training

:heavy_check_mark: Improved data structure

:heavy_check_mark: Different activation function for every layer

:heavy_check_mark: Fast saving and loading

# Quick start

Example project under `./SeayonDemo`

When you are using `./SeayonMnist` you have to download the `Mnist dataset` and put it into the `./SeayonMnist/res` folder.

# Including in your Project

Project is built using Visual Studio Code and CMake on Windows. The code runs about 30% faster with compiled with `GNU` than with `MSVC`.

The default branch contains the currently in development beta version. The latest stable version is found under the "Release" tab.
To use Seayon in your project you include the basis header:

```C++
#include "seayon.hpp"
```

To use the GPU accelerated version you include the cuda header and install the [Cuda toolkit](https://developer.nvidia.com/cuda-downloads). Follow the [Cuda Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows) to setup the compiler.

```C++
#include "cuda_seayon.cuh"
```

# Found a bug

If you have found an issue or you would like to submit improvements, use the **Issues tab**. The **Issues tab** can also be used to start discussions. Make sure beforehand whether the issue ticket has already been created.

# Contributing

I am very grateful for any contributions. Fork the repository to create your own copy, when you're done you can create a pull request.

Before opening a pull request, please submit an issue explaining the bug or feature and always reference the [issue hashtag](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/autolinked-references-and-urls#issues-and-pull-requests) in the discription with an [closing keyword](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue#linking-a-pull-request-to-an-issue-using-a-keyword): `close, closes, closed, fix, fixes, fixed, resolve, resolves, resolved`
