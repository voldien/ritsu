#pragma once
#include "Optimizer.h"
#include <functional>

namespace Ritsu {

	template <typename T> class SGD : public Optimizer<T> {
	  public:
		SGD(const T learningRate, const T momentum, const std::string &name = "SGD")
			: Optimizer<T>(learningRate, name) {
			this->momentum = momentum;
		}

		void update_step(Tensor &gradient, Tensor &variable) {
			if (momentum > 0) {
				velocity = momentum * velocity - gradient * this->getLearningRate();
				variable = variable + velocity;
			}
		}

	  private:
		T momentum;
		T velocity;
	};

} // namespace Ritsu