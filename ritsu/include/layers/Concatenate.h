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

		Tensor compute_derivative(const Tensor &tensorLoss) override {}
		Tensor &compute_derivative(Tensor &tensorLoss) const override {}

	  private:
		static void concatenate(const Tensor &tensorA, const Tensor &tensorB, Tensor &output) {
			Tensor copyA = tensorA;
			copyA.append(tensorB);
			output = copyA;
		}

		std::vector<Layer<DType> *> inputs;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu