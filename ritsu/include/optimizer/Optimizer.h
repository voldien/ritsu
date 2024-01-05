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

		/**
		 * @brief Set the Learning Rate object
		 * 
		 * @param rate 
		 */
		void setLearningRate(T rate) { this->learningRate = rate; }

		/**
		 * @brief Get the Learning Rate object
		 * 
		 * @return T 
		 */
		T getLearningRate() const { return this->learningRate; }

		/**
		 * @brief 
		 * 
		 * @param loss 
		 * @param variable 
		 * @param output_gradient 
		 */
		virtual void gradient(const Tensor &loss, const Tensor &variable, Tensor &output_gradient) {}

		/**
		 * @brief 
		 * 
		 * @param gradient 
		 * @param variable 
		 */
		virtual void update_step(const Tensor &gradient, Tensor &variable) {}

		/**
		 * @brief 
		 * 
		 * @param gradient 
		 * @param variable 
		 */
		virtual void apply(Tensor &gradient, Tensor &variable) {}

	  private:
		T learningRate;
		std::string name;
	};

} // namespace Ritsu