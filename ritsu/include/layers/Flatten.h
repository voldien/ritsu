#pragma once
#include "Layer.h"

namespace Ritsu {

	class Flatten : public Layer<float> {

		//  kernel_initializer='glorot_uniform',
		//    bias_initializer='zeros',

	  public:
		Flatten(Layer<float> &a, Layer<float> b) {}

	  private:
	};
} // namespace Ritsu