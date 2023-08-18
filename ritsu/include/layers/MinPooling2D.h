#pragma once
#include "Layer.h"

namespace Ritsu {
	class MinPooling2D : public Layer<float> {

	  public:
		MinPooling2D(int stride, const std::string &name = "minpooling") : Layer(name) {}

	  private:
		int stride;
	};
} // namespace Ritsu