#pragma once
#include "Layer.h"

namespace Ritsu {

	class Flatten : public Layer<float> {
	  public:
		Flatten(const std::string &name = "") : Layer<float>(name) {}

		

	  private:
	};
} // namespace Ritsu