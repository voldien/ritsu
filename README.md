# Ritsu - リツ
[![Ritsu](https://github.com/voldien/ritsu/actions/workflows/ci.yml/badge.svg)](https://github.com/voldien/ritsu/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub release](https://img.shields.io/github/release/voldien/ritsu.svg)](https://github.com/voldien/ritsu/releases)

logo
[]()

```cpp


```

## Supported Precisions

- **Float**
- **Double**
- **Integar**

## Layer Support

### Activation

- **Relu**
- **Leaky Relu**
- **Sigmoid**
- **Tanh**
- **SoftMax**
- **Linear**
- **Swish**
- **ExpLinear**
- **Linear**

### Layer
- **Dense**
- **Batch Normalization**
- **Dropout**
- **GuassianNoise**
- **Flatten**
- **Reshape**
- **Concatenate**
- **Add**
- **Subtract**
- **Multiply**
- **Divide**


## Loss Function
- **MSE** - Mean Square Error.
- **MSA** - Mean Square Absolute.
- **CrossEntropy** - 
- **CatagorialCrossEntropy** - 

## Optimizer

- **SGD** - Stochastic Gradient Descent
- **Adam** - Adaptive Moment estimation

## Dependencies

In order to compile the program on Linux based machine, the following Debian packages are required.

```bash
  sudo apt-get install cmake g++ libgtest-dev googletest libomp-dev libjemalloc-dev
```

## Building

### ROCM

### CUDA

### Intel

## Installation

## Testing

```bash
ctest -vv
```

```bash
ctest -O failure.txt --output-on-failure
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details