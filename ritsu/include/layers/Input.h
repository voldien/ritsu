#pragma once
#include "Layer.h"
#include <cmath>

namespace Ritsu {

	class Input : public Layer<float> {
	  public:
		Input(const std::vector<unsigned int> &input, const std::string &name = "input") : Layer<float>(name) {
			this->shape = input;
		}

	  private:
		// void setInput(const std::vector<Layer<DType> *> &layers) override { this->input = layers[0]; }
	};
} // namespace Ritsu