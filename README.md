# Ritsu - リツ

[![Ritsu](https://github.com/voldien/ritsu/actions/workflows/ci.yml/badge.svg)](https://github.com/voldien/ritsu/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub release](https://img.shields.io/github/release/voldien/ritsu.svg)](https://github.com/voldien/ritsu/releases)

A machine learning library, created for personal educational purposes. Is not recommended for product development. But only as educational material.

```cpp
Input input({2}, "input");
Dense dense0(2, false);
Dense outputDense(1, false);

RandomUniformInitializer<float> random(0, 2, 10052);
Tensor<float> dataX = random(Shape<unsigned int>({128, 2}));

/* Sum. */
Tensor<float> dataY({128, 1});
for (unsigned int i = 0; i < dataY.getNrElements(); i++) {

 const float value = dataX.getValue({i, 0}) + dataX.getValue({i, 1});
 dataY.getValue(i) = value;
}

Layer<float> &output = outputDense(dense0(input));
SGD<float> optimizer(0.0001, 0.0);

MetricAccuracy accuracy;
Model<float> forwardModel = Model<float>({&input}, {&output});
MeanSquareError mse_loss = MeanSquareError();
forwardModel.compile(&optimizer, mse_loss, {dynamic_cast<Metric *>(&accuracy)});

Model<float>::History *result = &forwardModel.fit(8, dataX, dataY, 1, 0, false, false);

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
- **Batch Normalization** - WIP
- **Dropout** - WIP
- **GuassianNoise** - WIP
- **Flatten**
- **Reshape**

## Loss Function

- **MSE** - Mean Square Error.
- **MSA** - Mean Square Absolute.
- **CrossEntropy** -
- **CatagorialCrossEntropy** -

## Optimizer

- **SGD** - Stochastic Gradient Descent
- **Adam** - Adaptive Moment Estimation

## Dependencies

In order to compile the program on Linux based machine, the following Debian packages are required.

```bash
  sudo apt-get install cmake g++ libgtest-dev googletest libomp-dev libjemalloc-dev
```

## Building

The OpenMP can be built with other graphic framework, to be executed on the graphic devices. However, no supported yet added.

### AMD - ROCM

### NVIDIA - CUDA

### Intel - OneAPI

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
