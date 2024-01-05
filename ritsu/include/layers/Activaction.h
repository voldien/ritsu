#pragma once
#include "Layer.h"

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class Activaction : public Layer<float> {
	  public:
		Activaction(const std::string &name = "activaction") : Layer(name) {}
		~Activaction() override {}

		// virtual Tensor operator<<(Tensor &tensor) override { return tensor; }
		//
		// virtual Tensor operator>>(Tensor &tensor) override { return tensor; }
		//
		// virtual Tensor &operator()(Tensor &tensor) override { return tensor; }
	};
} // namespace Ritsu