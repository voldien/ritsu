#pragma once
#include "Layer.h"

namespace Ritsu {

	/**
	 * @brief
	 *
	 * @tparam T
	 */
	template <typename T> class Cast : public Layer<T> {
	  public:
		Cast(const std::string &name = "cast") : Layer<T>(name){};

		template <class U> auto &operator()(U &layer) {

			// this->setInputs({&layer});
			// layer.setOutputs({this});

			return *this;
		}

		// void setInputs(const std::vector<Layer<T> *> &layers)  {}
		void setInputs(const std::vector<Layer<T> *> &layers) override {}
		void setOutputs(const std::vector<Layer<T> *> &layers) override {}

		Tensor compute_derivative(const Tensor &tensor) override { return tensor; }
		Tensor &compute_derivative(Tensor &tensor) const override { return tensor; }
	};
} // namespace Ritsu