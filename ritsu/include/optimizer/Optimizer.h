#pragma once
#include "Layer.h"

namespace Ritsu {

	template <typename T> class Optimizer {
	  public:
		void setLearningRate(T rate) {}

	  private:
		T learningRate;
	};

} // namespace Ritsu