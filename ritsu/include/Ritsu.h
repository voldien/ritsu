#include "Tensor.h"
#include "layers/Add.h"
#include "layers/AveragePooling.h"
#include "layers/BatchNormalization.h"
#include "layers/Cast.h"
#include "layers/Concatenate.h"
#include "layers/Conv2D.h"
#include "layers/Dense.h"
#include "layers/GaussianNoise.h"
#include "layers/Input.h"
#include "layers/Regulator.h"

#include "Model.h"
#include "layers/Flatten.h"
#include "layers/Layer.h"
#include "layers/MaxPooling.h"
#include "layers/Multiply.h"
#include "layers/Relu.h"
#include "layers/Sigmoid.h"
#include "layers/UpScale.h"
#include "optimizer/SGD.h"
// using TensorF = Tensor<float>;
// using TensorD = Tensor<double>;
// using TensorI = Tensor<int>;
// using TensorLI = Tensor<int>;