#pragma once
#include "Layer.h"

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class LambdaOp : public Layer<float> {
		using CustomLayerOp = void (*)(const Tensor<float> &input, const Tensor<float> &result);

	  public:
	};
} // namespace Ritsu