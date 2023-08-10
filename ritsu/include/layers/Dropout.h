#pragma once
#include "Layer.h"

namespace Ritsu {

	class Dropout : public Layer<float> {

	  public:
		Dropout(float perc, const std::string &name = "dropout") : Layer(name) {}

	  private:
	};
} // namespace Ritsu