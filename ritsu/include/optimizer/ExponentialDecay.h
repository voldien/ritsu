#pragma once
#include "Optimizer.h"
#include <functional>

namespace Ritsu {

	template <typename T> class ExponentialDecay {
	  public:
		ExponentialDecay(const T learningRate, const T decay_step, const T decay_rate) {
			this->learningRate = learningRate;
			this->decay_rate = decay_rate;
			this->decay_step = decay_step;
		}

	  private:
		T learningRate;
		T decay_step;
		T decay_rate;
	};
} // namespace Ritsu