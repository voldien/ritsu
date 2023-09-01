#pragma once
#include "Layer.h"

namespace Ritsu {
	class Rescaling : Layer<float> {
	  public:
		Rescaling(const DType scale, const std::string &name = "rescaling") : Layer<float>(name) {}
	};
} // namespace Ritsu