#pragma once
#include "Layer.h"

namespace Ritsu {

	/**
	 * @brief
	 *
	 * @tparam T
	 */
	template <typename T> class Multiply : public Layer<float> {

	  public:
		Multiply(const std::string &name = "multiply") : Layer<T>(name) {}

		Tensor<float> operator<<(const Tensor<float> &tensor) override { return tensor; }
		Tensor<float> &operator<<(Tensor<float> &tensor) override { return tensor; }
		Tensor<float> operator>>(Tensor<float> &tensor) override { return tensor; }
	
		void setInputs(const std::vector<Layer<DType> *> &layers) override {}
		void setOutputs(const std::vector<Layer<DType> *> &layers) override {}

		Tensor<float> compute_derivative(const Tensor<float> &tensor) override { return tensor; }
		Tensor<float> &compute_derivative(Tensor<float> &tensor) const override { return tensor; }

	  private:
	};
} // namespace Ritsu