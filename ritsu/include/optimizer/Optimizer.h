#pragma once
#include "../Tensor.h"
#include <string>

namespace Ritsu {

	/**
	 * @brief
	 *
	 * @tparam T
	 */
	template <typename T> class Optimizer {
	  public:
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		using DType = T;

	  public:
		Optimizer(T learningRate, const std::string &name) {
			this->setLearningRate(learningRate);
			this->name = name;
		}

		void setLearningRate(T rate) { this->learningRate = rate; }
		T getLearningRate() const { return this->learningRate; }

		virtual void gradient(const Tensor &loss, const Tensor &variable, Tensor &output_gradient) {}

		virtual void update_step(const Tensor &gradient, Tensor &variable) {}

		virtual void apply(Tensor &gradient, Tensor &variable) {}

	  private:
		T learningRate;
		std::string name;
	};

} // namespace Ritsu