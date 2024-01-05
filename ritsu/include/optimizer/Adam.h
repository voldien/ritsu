#pragma once
#include "Optimizer.h"
#include <functional>

namespace Ritsu {

	template <typename T> class Adam : public Optimizer<T> {
	  public:
		Adam(const T learningRate, const T beta_1, const T beta_2, const std::string &name = "adam")
			: Optimizer<T>(learningRate, name) {
			this->beta_1 = beta_1;
			this->beta_2 = beta_2;
		}

	  private:
		T beta_1;
		T beta_2;
	};

} // namespace Ritsu