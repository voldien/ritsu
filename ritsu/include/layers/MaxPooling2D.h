#pragma once
#include "Layer.h"

namespace Ritsu {

	class MaxPooling2D : public Layer<float> {

	  public:
		MaxPooling2D(int stride, const std::string &name = "maxpooling") : Layer(name) {}

	  private:
		int stride;
	};
} // namespace Ritsu