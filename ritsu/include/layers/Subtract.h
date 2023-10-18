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

		Tensor compute_derivative(const Tensor &tensor) override { return tensor; }
		Tensor &compute_derivative(Tensor &tensor) const override { return tensor; }

	  private:
		static inline void computeElementSum(Tensor &inputA, const Tensor &inputB) { inputA = inputA - inputB; }

	  private:
		std::vector<Layer<DType> *> inputs;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu