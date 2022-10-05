#pragma once
#include "Layer.h"

namespace Ritsu {

	class Add : public Layer<float> {

		//  kernel_initializer='glorot_uniform',
		//    bias_initializer='zeros',

	  public:
		Add(Layer<float> &a, Layer<float> b) {}

	  private:
	};
} // namespace Ritsu