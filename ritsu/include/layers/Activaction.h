#pragma once
#include "Layer.h"
#include "../Math.h"

namespace Ritsu {

	class Activaction : public Layer<float> {
	  public:
		Activaction(const std::string &name = "") : Layer(name) {}
		~Activaction() override {}

		// virtual Tensor operator<<(Tensor &tensor) override { return tensor; }
		//
		// virtual Tensor operator>>(Tensor &tensor) override { return tensor; }
		//
		// virtual Tensor &operator()(Tensor &tensor) override { return tensor; }
	};
} // namespace Ritsu