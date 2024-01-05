#pragma once
#include "Optimizer.h"
#include <functional>

namespace Ritsu {

	template <typename T> class Ada : public Optimizer<T> {
	  public:
		Ada(const T learningRate, T beta, const std::string &name = "ada") : Optimizer<T>(learningRate, name) {}
	};

} // namespace Ritsu