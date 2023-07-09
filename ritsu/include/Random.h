#pragma once

#include <random>
#include <string>
#include <vector>

namespace Ritsu {

	class Random {
	  public:
		Random() {
			std::random_device random_device;	// Will be used to obtain a seed for the random number engine
			std::mt19937 gen(random_device()); // Standard mersenne_twister_engine seeded with rd()
			std::uniform_real_distribution<> dis(1.0, 2.0);
		}
	};
} // namespace Ritsu