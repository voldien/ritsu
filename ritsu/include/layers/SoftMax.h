#pragma once
#include "../Activations.h"
#include "Activaction.h"
#include <cmath>

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class SoftMax : public Activaction {
	  public:
		SoftMax(const std::string &name = "softmax") : Activaction(name) {}
		~SoftMax() override {}

		Tensor<float> operator<<(const Tensor<float> &tensor) override {
			// compute(tensor);
			return tensor;
		}

		Tensor<float> operator>>(Tensor<float> &tensor) override {
			softMax<DType>(tensor);
			return tensor;
		}

		// virtual Tensor<float> operator()(Tensor<float> &tensor) {
		//	compute(tensor);
		//	return tensor;
		//}

		Tensor<float> &operator()(Tensor<float> &tensor) override {
			softMax<DType>(tensor);
			return tensor;
		}

		template <class U> auto &operator()(U &layer) {

			this->setInputs({&layer});
			layer.setOutputs({this});

			this->build(layer.getShape());

			return *this;
		}

		void build(const Shape<IndexType> &shape) override { this->shape = shape; }

		// virtual const Tensor<float> &operator()(const Tensor<float> &tensor) override {
		//	//compute(tensor);
		//	return tensor;
		//}

		void setInputs(const std::vector<Layer<DType> *> &layers) override {}
		void setOutputs(const std::vector<Layer<DType> *> &layers) override {}

		Tensor<float> compute_derivative(const Tensor<float> &tensor) override { return  tensor; }
		Tensor<float> &compute_derivative(Tensor<float> &tensor) const override { return tensor; }
	};
} // namespace Ritsu