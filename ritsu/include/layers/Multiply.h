#pragma once
#include "Layer.h"

namespace Ritsu {

	template <typename T> class Multiply : public Layer<float> {

	  public:
		Multiply(const std::string &name = "multiply") {}

		Multiply &operator<<(Multiply &layer) { return *this; }

		Multiply &operator>>(Multiply &layer) { return *this; }

		Multiply &operator()(Multiply &layer) { return *this; }

	  private:
	};
} // namespace Ritsu