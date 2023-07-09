#pragma once
#include "Optimizer.h"
#include <functional>

namespace Ritsu {

	template <typename T> class SGD : public Optimizer<T> {
	  public:
		SGD(T learningRate, T beta) : Optimizer<T>(learningRate) { this->beta = beta; }

	  private:
		T beta;
	};

} // namespace Ritsu