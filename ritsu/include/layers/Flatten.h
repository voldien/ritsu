#pragma once
#include "Layer.h"

namespace Ritsu {

	class Flatten : public Layer<float> {
	  public:
		Flatten(const std::string &name = "flatten") : Layer<float>(name) {}

		Tensor &operator()(Tensor &tensor) override { return tensor.flatten(); }

		Tensor &operator<<(Tensor &tensor) override { return tensor.flatten(); }

		Tensor operator<<(const Tensor &tensor) override {
			Tensor tmp = tensor;
			return tmp.flatten();
		}

		template <class U> auto &operator()(U &layer) {

			this->setInputs({&layer});
			layer.setOutputs({this});

			return *this;
		}

		void setInputs(const std::vector<Layer<DType> *> &layers) override {
			this->input = layers[0];

			this->shape = this->input->getShape().flatten();
		}

		void setOutputs(const std::vector<Layer<DType> *> &layers) override {
			/*	Set input layer */
			this->outputs = layers;
		}

		std::vector<Layer<DType> *> getInputs() const override { return {input}; }
		std::vector<Layer<DType> *> getOutputs() const override { return outputs; }

		Tensor compute_deriviate(const Tensor &tensor) override { return tensor; }
		Tensor &compute_deriviate(Tensor &tensor) const override { return tensor; }

	  private:
		Layer<DType> *input;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu