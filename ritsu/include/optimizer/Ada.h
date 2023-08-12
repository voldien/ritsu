#pragma once
#include "Optimizer.h"
#include <functional>

namespace Ritsu {

	template <typename T> class Ada : public Optimizer<T> {
	  public:
		Ada(T beta) {}
	};

} // namespace Ritsu