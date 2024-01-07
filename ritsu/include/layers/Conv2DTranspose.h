
#pragma once
#include "Layer.h"
#include <cstdint>
#include <ctime>
#include <vector>

namespace Ritsu {

	enum class ConvPadding { Same, Valid };
	class Conv2DTranspose : public Layer<float> {

		//  kernel_initializer='glorot_uniform',
		//    bias_initializer='zeros',

	  public:
		Conv2DTranspose(const size_t filters, const std::vector<uint32_t> kernel_size,
						const std::vector<uint32_t> strides = {1, 1}, const std::string &padding = "valid",
						const std::string name = "conv2Dtranspose")
			: Layer<float>(name) {}

		void setInputs(const std::vector<Layer<DType> *> &layers) override {}
		void setOutputs(const std::vector<Layer<DType> *> &layers) override {}

		Tensor compute_derivative(const Tensor &tensorLoss) override {}
		Tensor &compute_derivative(Tensor &tensorLoss) const override {}

	  private:
	};
} // namespace Ritsu