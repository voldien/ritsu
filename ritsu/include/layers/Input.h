#pragma once
#include "Layer.h"
#include <cmath>

namespace Ritsu {

	class Input : public Layer<float> {
	  public:
		Input(const std::vector<unsigned int> &input, const std::string &name = "input") : Layer<float>(name) {
			this->shape = input;
		}

		void setInputs(const std::vector<Layer<DType> *> &layers) override {}

		void setOutputs(const std::vector<Layer<DType> *> &layers) override {
			/*	Set input layer */
			this->outputs = layers;
		}

		std::vector<Layer<DType> *> getInputs() const override { return {}; }
		std::vector<Layer<DType> *> getOutputs() const override { return outputs; }

		Tensor compute_deriviate(const Tensor &tensor) override { return tensor; }
		Tensor &compute_deriviate(Tensor &tensor) const override { return tensor; }

	  private:
		std::vector<Layer<DType> *> outputs;
		// void setInput(const std::vector<Layer<DType> *> &layers) override { this->input = layers[0]; }
	};
} // namespace Ritsu