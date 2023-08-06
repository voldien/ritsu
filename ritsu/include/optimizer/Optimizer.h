#pragma once
#include "layers/Layer.h"

namespace Ritsu {

	template <typename T> class Optimizer {
	  public:
		using DType = T;

	  public:
		Optimizer(T learningRate, const std::string &name) {
			this->setLearningRate(learningRate);
			this->name = name;
		}

		void setLearningRate(T rate) { this->learningRate = rate; }
		T getLearningRate() const { return this->learningRate; }

		virtual void update_step(float* gradient, float* variable){}

	  private:
		T learningRate;
		std::string name;
	};

} // namespace Ritsu