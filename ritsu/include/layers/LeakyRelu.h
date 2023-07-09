#pragma once
#include "Layer.h"
#include <ctime>
#include <random>

namespace Ritsu {

	class LeakyRelu : public Layer<float> {
	  public:
		LeakyRelu();
	};
} // namespace Ritsu