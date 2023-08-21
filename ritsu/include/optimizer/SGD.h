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

		void gradient(const Tensor &loss, const Tensor &variable, Tensor &output_gradient) override {}

		void update_step(const Tensor &gradient, Tensor &variable) override {
			Tensor tmpGradient = gradient;
			if (momentum > 0) {
				// velocity = momentum * velocity - (gradient * this->getLearningRate());
				// variable = variable + velocity;
			} else {
				Tensor gradientUpdate = tmpGradient * this->getLearningRate();

				if (gradientUpdate.getShape() == variable.getShape()) {
					// assert(train_variables->getShape() == differental_gradient.getShape());
					variable = gradientUpdate;
				}
			}
		}

	  private:
		T momentum;
		T velocity;
	};

} // namespace Ritsu