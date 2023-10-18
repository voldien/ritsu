#pragma once
#include "Layer.h"

namespace Ritsu {

	/**
	 * @brief Scale value.
	 *
	 */
	class Rescaling : public Layer<float> {
	  public:
		Rescaling(const DType scale, const std::string &name = "rescaling") : Layer<float>(name) {
			this->scale = scale;
		}

		Tensor &operator()(Tensor &tensor) override {
			this->computeScale(tensor);
			return tensor;
		}

		Tensor &operator<<(Tensor &tensor) override {
			this->computeScale(tensor);
			return tensor;
		}

		Tensor operator<<(const Tensor &tensor) override {
			Tensor tmp = tensor;
			this->computeScale(tmp);
			return tmp;
		}

		template <class U> auto &operator()(U &layer) {

			this->setInputs({&layer});
			layer.setOutputs({this});

			return *this;
		}

		void setInputs(const std::vector<Layer<DType> *> &layers) override {
			this->input = layers[0];

			this->shape = this->input->getShape();
		}

		void setOutputs(const std::vector<Layer<DType> *> &layers) override {
			/*	Set input layer */
			this->outputs = layers;
		}

		std::vector<Layer<DType> *> getInputs() const override { return {input}; }
		std::vector<Layer<DType> *> getOutputs() const override { return outputs; }

		Tensor compute_derivative(const Tensor &tensor) override { return tensor; }
		Tensor &compute_derivative(Tensor &tensor) const override { return tensor; }

	  protected:
		void computeScale(Tensor &tensor) {
			for (size_t i = 0; i < tensor.getNrElements(); i++) {
				tensor.getValue<DType>(i) = scale * tensor.getValue<DType>(i);
			}
		}

	  private:
		DType scale;
		Layer<DType> *input;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu