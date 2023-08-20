#pragma once
#include "Layer.h"
#include <cstdint>

namespace Ritsu {

	class UpScale : public Layer<float> {

	  public:
		UpScale(uint32_t scale, const std::string &name = "upscale") : Layer(name) {}

	  private:
		// void compute(Tensor &a, Tensor &b) { return a + b; }

		std::vector<Layer<DType> *> inputs;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu