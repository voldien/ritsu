#pragma once
#include "Add.h"
#include "Layer.h"
#include <string>

namespace Ritsu {

	class Subtract : public Add {

	  public:
		Subtract(const std::string &name = "subtract") {}

	  private:
		static inline void computeElementSum(Tensor &inputA, const Tensor &inputB) { inputA = inputA - inputB; }
	};
} // namespace Ritsu