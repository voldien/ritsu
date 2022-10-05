#pragma once
#include"../Tensor.h"
#include <omp.h>
#include <string>
#include <vector>

namespace Ritsu {

	template <typename T> class Layer {

	  public:
		using DType = T;
		Layer(const std::string &name) {}


		virtual Tensor operator<<(Tensor &tensor) { return tensor; }

	
		virtual Tensor operator>>(Tensor &tensor) { return tensor; }


		virtual Tensor &operator()(Tensor &tensor) { return tensor; }

		// Dtype

		// Weights trainable

		// non-trainable.

		// input
		std::vector<Layer<T>> getInputs() const;

		// name

		// output
		std::vector<Layer<T>> getOutputs() const;

		// trainable.

	  private:
	};
} // namespace Ritsu