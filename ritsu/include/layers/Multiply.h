#pragma once
#include "Layer.h"

namespace Ritsu {

	template <typename T> class Multiply : public Layer<float> {

	  public:
		using DType = T;
		Multiply(const std::string &name);

		Multiply &operator<<(Multiply &layer) { return *this; }

		Multiply &operator>>(Multiply &layer) { return *this; }

		Multiply &operator()(Multiply &layer) { return *this; }

		// Dtype

		// Weights trainable
		// non-trainable.

		// input

		// name

		// output

		// trainable.

	  private:
	};
} // namespace Ritsu