#pragma once
#include "Tensor.h"
#include <functional>

namespace Ritsu {

	class Loss {

	  public:
		using LossFunction = void (*)(const Tensor &, const Tensor &, Tensor &);

		Loss() = default;
		//	template <typename T>
		Loss(LossFunction lambda, const std::string &name = "loss") : name(name) { this->loss_function = lambda; }
		virtual Tensor computeLoss(const Tensor &inputX0, const Tensor &inputX1) {
			Tensor out;

			Tensor::verifyShape(inputX0, inputX1);

			this->loss_function(inputX0, inputX1, out);
			return out;
		}

	  private:
		LossFunction loss_function;
		std::string name;
	};
}; // namespace Ritsu