#pragma once
#include "Layer.h"

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class LambdaOp : public Layer<float> {
		using CustomLayerOp = void (*)(const Tensor<float> &input, const Tensor<float> &result);

		Tensor<DType> &call(Tensor<DType> &tensor, bool training) override { return tensor; }

		Tensor<DType> call(const Tensor<DType> &tensor, bool training) override {
			Tensor<float> output({1});
			return output;
		}

	  public:
	};
} // namespace Ritsu