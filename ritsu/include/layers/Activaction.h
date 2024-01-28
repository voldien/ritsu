#pragma once
#include "Activations.h"
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

		// virtual Tensor<float> operator<<(Tensor<float> &tensor) override { return tensor; }
		//
		// virtual Tensor<float> operator>>(Tensor<float> &tensor) override { return tensor; }
		//
		// virtual Tensor<float> &operator()(Tensor<float> &tensor) override { return tensor; }
	};
} // namespace Ritsu