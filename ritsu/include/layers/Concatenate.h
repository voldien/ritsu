#pragma once
#include "Layer.h"

namespace Ritsu {

	class Concatenate : public Layer<float> {

	  public:
		Concatenate(Layer<DType> &a, Layer<DType> b, const std::string &name) : Concatenate({&a, &b}, name) {}
		Concatenate(const std::vector<Layer<DType> *> &layers, const std::string &name) : Layer<float>(name) {
			this->inputs = layers;
		}

	  private:
		// void compute(Tensor &a, Tensor &b) { return a + b; }

		std::vector<Layer<DType> *> inputs;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu