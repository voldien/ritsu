#pragma once
#include "Layer.h"

namespace Ritsu {

	class MaxPooling : public Layer<float> {

	  public:
		MaxPooling(int stride, const std::string &name = "") : Layer(name) {}

	  private:
		int stride;
	};
} // namespace Ritsu