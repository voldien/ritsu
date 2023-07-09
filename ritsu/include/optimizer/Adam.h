#pragma once
#include "Optimizer.h"
#include <functional>

namespace Ritsu {

	template <typename T> class Adam : public Optimizer<T> {
	  public:
		Adam(T beta) {}
	};

} // namespace Ritsu