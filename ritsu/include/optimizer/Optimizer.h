#pragma once
#include "layers/Layer.h"

namespace Ritsu {

	template <typename T> class Optimizer {
	  public:
		using DType = T;

	  public:
		Optimizer(T learningRate) { this->setLearningRate(learningRate); }

		void setLearningRate(T rate) { this->learningRate = rate; }
		T getLearningRate() const { return this->learningRate; }

	  private:
		T learningRate;
	};

} // namespace Ritsu