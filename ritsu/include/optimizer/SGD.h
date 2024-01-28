#pragma once
#include "Optimizer.h"
#include <cassert>
#include <functional>

namespace Ritsu {

	/**
	 * @brief
	 *
	 * @tparam T
	 */
	template <typename T> class SGD : public Optimizer<T> {
	  public:
		SGD(const T learningRate, const T momentum = 0.0f, const std::string &name = "SGD")
			: Optimizer<T>(learningRate, name) {
			this->momentum = momentum;
		}

		void gradient(const Tensor<float> &loss, const Tensor<float> &variable, Tensor<float> &output_gradient) override {}

		void update_step(const Tensor<float> &gradient, Tensor<float> &variable) override {

			Tensor<float> tmpGradient = gradient;
			
			if (momentum > 0) {
				// velocity = momentum * velocity - (gradient * this->getLearningRate());
				// variable = variable + velocity;
			} else {
				Tensor<float> gradientUpdate = tmpGradient * this->getLearningRate();

				/*	*/
				std::cout << gradientUpdate.getShape() << " " << variable.getShape() << std::flush;
				
				// TODO: check and validate.
				assert(gradientUpdate.getShape() == variable.getShape());
				if (gradientUpdate.getShape() == variable.getShape()) {
					// ;
					variable = gradientUpdate;
				}
			}
		}

	  private:
		T momentum;
		T velocity;
	};

} // namespace Ritsu