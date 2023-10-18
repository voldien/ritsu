/*
 * Copyright (c) 2023 Valdemar Lindberg
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
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
#include "layers/LeakyRelu.h"
#include "layers/Linear.h"
#include "layers/MinPooling2D.h"
#include "layers/Regularization.h"
#include "layers/Rescaling.h"
#include "layers/Reshape.h"
#include "layers/Swish.h"
#include "layers/UpScale.h"

#include "Metric.h"
#include "Model.h"
#include "layers/Flatten.h"
#include "layers/Layer.h"
#include "layers/MaxPooling2D.h"
#include "layers/Multiply.h"
#include "layers/Relu.h"
#include "layers/Sigmoid.h"
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
