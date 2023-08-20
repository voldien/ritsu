#pragma once
#include "Layer.h"

namespace Ritsu {

	class Dropout : public Layer<float> {

	  public:
		Dropout(const DType perc, const std::string &name = "dropout") : Layer(name), perc(perc) {}

		Tensor &operator<<(Tensor &tensor) override {
			this->computeDropout(tensor);
			return tensor;
		}

		Tensor operator<<(const Tensor &tensor) override {
			Tensor tmpOutput = tensor;
			this->computeDropout(tmpOutput);
			return tmpOutput;
		}

		std::vector<Layer<DType> *> getInputs() const override { return {input}; }
		std::vector<Layer<DType> *> getOutputs() const override { return outputs; }

		void build(const Shape<IndexType> &shape) override {}

		Tensor compute_derivative(const Tensor &tensorLoss) override { return tensorLoss; }
		Tensor &compute_derivative(Tensor &tensorLoss) const override { return tensorLoss; }

	  private:
		void computeDropout(Tensor &tensor) { /*	Iterate through each all elements.    */

			/*	*/
		}

		/*	*/
		DType perc;
		Layer<DType> *input;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu