#pragma once
#include "Layer.h"

namespace Ritsu {

	class UpScale : public Layer<float> {

	  public:
		UpScale(float scale, const std::string &name) : Layer(name) {}

	  private:
		// void compute(Tensor &a, Tensor &b) { return a + b; }

		std::vector<Layer<DType> *> inputs;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu