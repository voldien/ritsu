#pragma once
#include "Add.h"
#include "Layer.h"
#include <string>

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class Subtract : public Add {
	  public:
		Subtract(const std::string &name = "subtract") {}

		void setInputs(const std::vector<Layer<DType> *> &layers) override { this->inputs = layers; }
		void setOutputs(const std::vector<Layer<DType> *> &layers) override { this->outputs = layers; }

		Tensor<float> compute_derivative(const Tensor<float> &tensor) override { return tensor; }
		Tensor<float> &compute_derivative(Tensor<float> &tensor) const override { return tensor; }

	  private:
		static inline void computeElementSum(Tensor<float> &inputA, const Tensor<float> &inputB) { inputA = inputA - inputB; }

	  private:
		std::vector<Layer<DType> *> inputs;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu