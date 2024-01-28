#pragma once
#include "Layer.h"
#include "Tensor.h"

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class Concatenate : public Layer<float> {

	  public:
		Concatenate(Layer<DType> &a, Layer<DType> &b, const std::string &name = "concatenate")
			: Concatenate({&a, &b}, name) {}
		Concatenate(const std::vector<Layer<DType> *> &layers, const std::string &name) : Layer<float>(name) {
			this->inputs = layers;
		}

		template <class U> auto &operator()(U &layer) {

			this->setInputs({&layer});
			layer.setOutputs({this});

			this->build(layer.getShape());

			return *this;
		}

		void setInputs(const std::vector<Layer<DType> *> &layers) override {}
		void setOutputs(const std::vector<Layer<DType> *> &layers) override {}

		Tensor<float> compute_derivative(const Tensor<float> &tensorLoss) override {			return tensorLoss;}
		Tensor<float> &compute_derivative(Tensor<float> &tensorLoss) const override {			return tensorLoss;}

	  private:
		static void concatenate(const Tensor<float> &tensorA, const Tensor<float> &tensorB, Tensor<float> &output) {
			Tensor<float> copyA = tensorA;
			copyA.append(tensorB);
			output = copyA;
		}

		std::vector<Layer<DType> *> inputs;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu