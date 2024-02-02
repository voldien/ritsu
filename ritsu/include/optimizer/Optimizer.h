#pragma once
#include "../Tensor.h"
#include "Object.h"
#include <string>

namespace Ritsu {

	/**
	 * @brief
	 *
	 * @tparam T
	 */
	template <typename T> class Optimizer : Object {
	  public:
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");
		using DType = T;

	  public:
		Optimizer(T learningRate, const std::string &name) : Object(name) { this->setLearningRate(learningRate); }

		/**
		 * @brief Set the Learning Rate object
		 *
		 * @param rate
		 */
		virtual void setLearningRate(T rate) noexcept { this->learningRate = rate; }

		/**
		 * @brief Get the Learning Rate object
		 *
		 * @return T
		 */
		virtual T getLearningRate() const noexcept { return this->learningRate; }

		/**
		 * @brief
		 *
		 * @param loss
		 * @param variable
		 * @param output_gradient
		 */
		virtual void gradient(const Tensor<float> &loss, const Tensor<float> &variable,
							  Tensor<float> &output_gradient) {}

		/**
		 * @brief
		 *
		 * @param gradient
		 * @param variable
		 */
		virtual void update_step(const Tensor<float> &gradient, Tensor<float> &variable) {}

		/**
		 * @brief
		 *
		 * @param gradient
		 * @param variable
		 */
		virtual void apply(Tensor<float> &gradient, Tensor<float> &variable) {}

	  private:
		T learningRate;
		std::string name;
	};

} // namespace Ritsu