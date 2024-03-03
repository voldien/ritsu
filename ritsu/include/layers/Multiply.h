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
	
		void build(const Shape<IndexType> &shape) override { this->shape = shape; }

		void setInputs(const std::vector<Layer<DType> *> &layers) override {
			if (layers.size() <= 1) {
				throw InvalidArgumentException("Must be greater or equal 2 layers");
			}
			// Check if shape is valid.
			for (size_t i = 0; i < layers.size(); i++) {
			}
			this->inputs = layers;
			this->shape = layers[0]->getShape();
		}
		void setOutputs(const std::vector<Layer<DType> *> &layers) override { this->outputs = layers; }


		Tensor<float> compute_derivative(const Tensor<float> &tensor) override { return tensor; }
		Tensor<float> &compute_derivative(Tensor<float> &tensor) const override { return tensor; }

	  protected:
		std::vector<Layer<DType> *> inputs;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu