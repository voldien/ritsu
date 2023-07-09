#pragma once
#include "Tensor.h"

namespace Ritsu {

	class Loss {
	  public:
		virtual Tensor computeLoss(const Tensor &X0, const  Tensor &X1) { return {}; }
	};
}; // namespace Ritsu