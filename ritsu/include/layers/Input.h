#pragma once
#include "Layer.h"

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class Input : public Layer<float> {
	  public:
		Input(const std::vector<IndexType> &input, const std::string &name = "input") : Layer<float>(name) {
			this->shape = input;
		}

		void setInputs(const std::vector<Layer<DType> *> &layers) override {
			/*	No input layer connection, since input layer.	*/
		}

		void setOutputs(const std::vector<Layer<DType> *> &layers) override {
			/*	Set input layer */
			this->outputs = layers;
		}

		std::vector<Layer<DType> *> getInputs() const override { return {}; }
		std::vector<Layer<DType> *> getOutputs() const override { return outputs; }

		Tensor<float> compute_derivative(const Tensor<float> &tensor) override { return tensor; }
		Tensor<float> &compute_derivative(Tensor<float> &tensor) const override { return tensor; }

	  private:
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu