#pragma once
#include "Optimizer.h"
#include <functional>

namespace Ritsu {

	template <typename T> class Adam : public Optimizer<T> {
	  public:
		Adam(const T learningRate, const T beta, const std::string &name = "") : Optimizer<T>(learningRate, name) {}
	};

} // namespace Ritsu