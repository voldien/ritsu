#pragma once
#include "Layer.h"

namespace Ritsu {

	/**
	 * @brief 
	 * 
	 * @tparam T 
	 */
	template <typename T> class Multiply : public Layer<float> {

	  public:
		Multiply(const std::string &name = "multiply") {}

		Tensor operator<<(const Tensor &tensor) override { return tensor; }

		Tensor &operator<<(Tensor &tensor) override { return tensor; }

		Tensor operator>>(Tensor &tensor) override { return tensor; }

		Tensor &operator()(Tensor &tensor) override { return tensor; }

	  private:
	};
} // namespace Ritsu