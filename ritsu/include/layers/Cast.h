#pragma once
#include "Layer.h"

namespace Ritsu {

	template <typename T> class Cast : public Layer<T> {
	  public:
		Cast(const std::string &name = "") : Layer<T>(name){};
	};
} // namespace Ritsu