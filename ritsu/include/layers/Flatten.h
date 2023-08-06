#pragma once
#include "Layer.h"

namespace Ritsu {

	class Flatten : public Layer<float> {
	  public:
		Flatten(const std::string &name = "flatten") : Layer<float>(name) {}

		Tensor &operator()(Tensor &tensor) override { return tensor.flatten(); }

		template <class U> auto &operator()(U &layer) {

			this->setInputs({&layer});
			layer.setOutputs({this});

			return *this;
		}

	  private:
	};
} // namespace Ritsu