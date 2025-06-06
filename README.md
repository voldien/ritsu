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

## Example Run Image Recognition MNIST

```bash
bin/mnist -B 16 -E 8 -l 0.00000001
Loaded MNIST Data Set: [60000,28,28,1] Labels: [60000,1]
Train Object Size: [28,28,1] Expected result Size: [10]
input    [28,28,1]
flatten0         [784]  <-- [ input ] f
layer0   [32]   <-- [ flatten0 ] f
relu0    [32]   <-- [ layer0 ] f
layer1   [16]   <-- [ relu0 ] f
relu1    [16]   <-- [ layer1 ] f
layer2   [10]   <-- [ relu1 ] f
number of weights: 25818
Trainable in Bytes: 100 KB
None-Trainable in Bytes: 0 KB
Loss Function: Categorical Crossentropy
Optimizer: sgd

Epoch: 0 / 8
Batch: 3375/3375 ETA: 0 - lr: 1e-08 loss: 0.00194646 - accuracy: 0.3
Batch: 375/375 ETA:  loss-val: 0.00136129 - accuracy: 0.25625
Epoch: 1 / 8
Batch: 3375/3375 ETA: 0 - lr: 1e-08 loss: 0.00134664 - accuracy: 0.475
Batch: 375/375 ETA:  loss-val: 0.00102787 - accuracy: 0.38125
Epoch: 2 / 8
Batch: 3375/3375 ETA: 0 - lr: 1e-08 loss: 0.00113783 - accuracy: 0.5625
Batch: 375/375 ETA:  loss-val: 0.000954269 - accuracy: 0.4625
Epoch: 3 / 8
Batch: 3375/3375 ETA: 0 - lr: 1e-08 loss: 0.0010338 - accuracy: 0.58125
Batch: 375/375 ETA:  loss-val: 0.000924363 - accuracy: 0.525
Epoch: 4 / 8
Batch: 3375/3375 ETA: 0 - lr: 1e-08 loss: 0.000975035 - accuracy: 0.60625
Batch: 375/375 ETA:  loss-val: 0.00090925 - accuracy: 0.54375
Epoch: 5 / 8
Batch: 3375/3375 ETA: 0 - lr: 1e-08 loss: 0.00093868 - accuracy: 0.61875
Batch: 375/375 ETA:  loss-val: 0.000900569 - accuracy: 0.5625
Epoch: 6 / 8
Batch: 3375/3375 ETA: 0 - lr: 1e-08 loss: 0.000916001 - accuracy: 0.64375
Batch: 375/375 ETA:  loss-val: 0.000894858 - accuracy: 0.5875
Epoch: 7 / 8
Batch: 3375/3375 ETA: 0 - lr: 1e-08 loss: 0.000901652 - accuracy: 0.6625
Batch: 375/375 ETA:  loss-val: 0.000890853 - accuracy: 0.59375

Average Test Loss: 0.00201005

Accuracy Test: 0.67083,
```

## Example Run AutoEncoder MNIST

```bash
bin/autoencoder -E 8 -B 1 -l 0.00001
Loaded MNIST Data Set: [60000,28,28,1] Labels: [60000,1]
Train Object Size: [28,28,1] Expected result Size: [28,28,1]
input    [28,28,1]
flatten0         [784]  <-- [ input ] f
layer0   [128]  <-- [ flatten0 ] f
relu0    [128]  <-- [ layer0 ] f
layer1   [64]   <-- [ relu0 ] f
relu1    [64]   <-- [ layer1 ] f
layer2   [32]   <-- [ relu1 ] f
relu2    [32]   <-- [ layer2 ] f
latent   [8]    <-- [ relu2 ] f
layer3   [32]   <-- [ latent ] f
relu3    [32]   <-- [ layer3 ] f
layer4   [64]   <-- [ relu3 ] f
relu4    [64]   <-- [ layer4 ] f
layer5   [128]  <-- [ relu4 ] f
relu5    [128]  <-- [ layer5 ] f
layer6   [784]  <-- [ relu5 ] f
sigmoid  [784]  <-- [ layer6 ] f
reshape  [28,28,1]      <-- [ sigmoid ] f
number of weights: 222936
Trainable in Bytes: 870 KB
None-Trainable in Bytes: 0 KB
Loss Function: MSE
Optimizer: adam
Epoch: 0 / 8
Batch: 54000/54000 ETA: 0 - lr: 1e-05 loss: 0.29203 - accuracy: 0.207908
Batch: 6000/6000 ETA:  loss-val: 0.297233 - accuracy: 0.190051
Epoch: 1 / 8
Batch: 54000/54000 ETA: 0 - lr: 1e-05 loss: 0.264627 - accuracy: 0.357143
Batch: 6000/6000 ETA:  loss-val: 0.269659 - accuracy: 0.331633
Epoch: 2 / 8
Batch: 54000/54000 ETA: 0 - lr: 1e-05 loss: 0.240133 - accuracy: 0.459184
Batch: 6000/6000 ETA:  loss-val: 0.244998 - accuracy: 0.438776
Epoch: 3 / 8
Batch: 54000/54000 ETA: 0 - lr: 1e-05 loss: 0.218462 - accuracy: 0.571429
Batch: 6000/6000 ETA:  loss-val: 0.223162 - accuracy: 0.561224
Epoch: 4 / 8
Batch: 54000/54000 ETA: 0 - lr: 1e-05 loss: 0.199439 - accuracy: 0.665816
Batch: 6000/6000 ETA:  loss-val: 0.203973 - accuracy: 0.655612
Epoch: 5 / 8
Batch: 54000/54000 ETA: 0 - lr: 1e-05 loss: 0.182833 - accuracy: 0.778061
Batch: 6000/6000 ETA:  loss-val: 0.187198 - accuracy: 0.765306
Epoch: 6 / 8
Batch: 54000/54000 ETA: 0 - lr: 1e-05 loss: 0.168387 - accuracy: 0.817602
Batch: 6000/6000 ETA:  loss-val: 0.172584 - accuracy: 0.797194
Epoch: 7 / 8
Batch: 54000/54000 ETA: 0 - lr: 1e-05 loss: 0.155844 - accuracy: 0.821429
Batch: 6000/6000 ETA:  loss-val: 0.159869 - accuracy: 0.808673
Average Test Loss: 0.16282
Accuracy Test: 0.828147
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

- **SGD** - Stochastic Gradient Descent + Momentum
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

## Binary Analysis

Extract disassembled unmanaged debug info.

```bash
objdump -CDS binary.elf
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## MNIST DataSet

<https://www.kaggle.com/datasets/hojjatk/mnist-dataset>

<http://yann.lecun.com/exdb/mnist/>

