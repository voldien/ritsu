#pragma once
#include "Layer.h"

namespace Ritsu {

	class Dropout : public Layer<float> {

	  public:
		Dropout(float perc, const std::string &name = "dropout") : Layer(name) {}

		Tensor &operator<<(Tensor &tensor) override {
			this->computeDropout(tensor);
			return tensor;
		}

		Tensor operator<<(const Tensor &tensor) override {
			Tensor tmp = tensor;
			this->computeDropout(tmp);
			return tmp;
		}

		std::vector<Layer<DType> *> getInputs() const override { return {input}; }
		std::vector<Layer<DType> *> getOutputs() const override { return outputs; }

		Tensor compute_derivative(const Tensor &tensor) override { return tensor; }
		Tensor &compute_derivative(Tensor &tensor) const override { return tensor; }

	  private:
		void computeDropout(Tensor &tensor) {
			/*Iterate through each all elements.    */

		}

		/*	*/
		Layer<DType> *input;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu