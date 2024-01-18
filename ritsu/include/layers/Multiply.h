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
		Multiply(const std::string &name = "multiply") : Layer<T>(name) {}

		Tensor operator<<(const Tensor &tensor) override { return tensor; }
		Tensor &operator<<(Tensor &tensor) override { return tensor; }
		Tensor operator>>(Tensor &tensor) override { return tensor; }
		Tensor &operator()(Tensor &tensor) override { return tensor; }

		template <class U> auto &operator()(U &layer) {

			this->setInputs({&layer});
			layer.setOutputs({this});

			this->build(layer.getShape());

			return *this;
		}

		void setInputs(const std::vector<Layer<DType> *> &layers) override {}
		void setOutputs(const std::vector<Layer<DType> *> &layers) override {}

		Tensor compute_derivative(const Tensor &tensor) override { return tensor; }
		Tensor &compute_derivative(Tensor &tensor) const override { return tensor; }

	  private:
	};
} // namespace Ritsu