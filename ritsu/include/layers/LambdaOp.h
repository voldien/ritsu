#pragma once
#include "Layer.h"

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class LambdaOp : public Layer<float> {
		using CustomLayerOp = void (*)(const Tensor &input, const Tensor &result);

	  public:
	};
} // namespace Ritsu