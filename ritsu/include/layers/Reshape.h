#pragma once
#include "Layer.h"
#include "core/Shape.h"

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class Reshape : public Layer<float> {
	  public:
		Reshape(const Shape<IndexType> &shape, const std::string &name = "reshape")
			: Layer<float>(name), newShape(shape) {}

		Tensor<float> &operator()(Tensor<float> &tensor) override {
			tensor.reshape(this->newShape);
			return tensor;
		}

		Tensor<float> &operator<<(Tensor<float> &tensor) override {
			tensor.reshape(this->newShape);
			return tensor;
		}

		Tensor<float> operator<<(const Tensor<float> &tensor) override {
			Tensor<float> tmp = tensor;
			tmp.reshape(this->newShape);
			return tmp;
		}

		template <class U> auto &operator()(U &layer) {

			this->setInputs({&layer});
			layer.setOutputs({this});

			return *this;
		}

		void setInputs(const std::vector<Layer<DType> *> &layers) override {

			Layer<DType> *layer = layers[0];
			/*	*/
			if (layer->getShape().getNrElements() != this->getShape().getNrElements()) {
				/*	*/
			}

			this->input = layers[0];
		}

		void setOutputs(const std::vector<Layer<DType> *> &layers) override {
			/*	Set input layer */
			this->outputs = layers;
		}

		std::vector<Layer<DType> *> getInputs() const override { return {input}; }
		std::vector<Layer<DType> *> getOutputs() const override { return outputs; }

		Tensor<float> compute_derivative(const Tensor<float> &tensor) override { return tensor; }
		Tensor<float> &compute_derivative(Tensor<float> &tensor) const override { return tensor; }

	  private:
		Layer<DType> *input;
		std::vector<Layer<DType> *> outputs;
		Shape<IndexType> newShape;
	};
} // namespace Ritsu