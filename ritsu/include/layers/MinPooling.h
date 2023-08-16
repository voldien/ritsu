#pragma once
#include "Layer.h"

namespace Ritsu {
	class MinPooling : public Layer<float> {

	  public:
		MinPooling(int stride, const std::string &name = "minpooling") : Layer(name) {}

	  private:
		int stride;
	};
} // namespace Ritsu