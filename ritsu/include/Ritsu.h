#pragma once
#include <cstddef>

#include "Tensor.h"
#include "layers/Add.h"
#include "layers/AveragePooling2D.h"
#include "layers/BatchNormalization.h"
#include "layers/Cast.h"
#include "layers/Concatenate.h"
#include "layers/Conv2D.h"
#include "layers/Dense.h"
#include "layers/ExpLinear.h"
#include "layers/GaussianNoise.h"
#include "layers/Input.h"
#include "layers/Linear.h"
#include "layers/MinPooling2D.h"
#include "layers/Regularization.h"
#include "layers/Reshape.h"
#include "layers/Swish.h"

#include "Metric.h"
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

namespace Ritsu {

	// using TensorF = Tensor<float>;
	// using TensorD = Tensor<double>;
	// using TensorI = Tensor<std::int32_t>;
	// using TensorI = Tensor<std::uint32_t>;
	// using TensorLI = Tensor<std::int64_t>;

}
