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