#pragma once
#include "Layer.h"

namespace Ritsu {

	template <typename T> class Cast : public Layer<T> {
	  public:
		Cast(const std::string &name = "Cast") : Layer<T>(name){};

		void setInputs(const std::vector<Layer<T> *> &layers) override {}
		void setOutputs(const std::vector<Layer<T> *> &layers) override {}

		Tensor compute_deriviate(const Tensor &tensor) override { return tensor; }
		Tensor &compute_deriviate(Tensor &tensor) const override { return tensor; }
	};
} // namespace Ritsu