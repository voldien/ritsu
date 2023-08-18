#pragma once
#include "Tensor.h"
#include "layers/Add.h"
#include "layers/AveragePooling2D.h"
#include "layers/BatchNormalization.h"
#include "layers/Cast.h"
#include "layers/Concatenate.h"
#include "layers/Conv2D.h"
#include "layers/Dense.h"
#include "layers/GaussianNoise.h"
#include "layers/Input.h"
#include "layers/MinPooling2D.h"
#include "layers/Regularization.h"

#include "Model.h"
#include "layers/Flatten.h"
#include "layers/Layer.h"
#include "layers/MaxPooling2D.h"
#include "layers/Multiply.h"
#include "layers/Relu.h"
#include "layers/Sigmoid.h"
#include "layers/UpScale.h"
#include "optimizer/Ada.h"
#include "optimizer/Adam.h"
#include "optimizer/SGD.h"
// using TensorF = Tensor<float>;
// using TensorD = Tensor<double>;
// using TensorI = Tensor<int>;
// using TensorLI = Tensor<int>;