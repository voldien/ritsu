
#pragma once
#include "Layer.h"
#include <cstdint>
#include <ctime>
#include <vector>

namespace Ritsu {

	class Conv2DTranspose : public Layer<float> {

		//  kernel_initializer='glorot_uniform',
		//    bias_initializer='zeros',

	  public:
		Conv2DTranspose(size_t filters, std::vector<uint32_t> kernel_size, const std::vector<uint32_t> strides = {1, 1},
						const std::string &padding = "valid", const std::string name = "conv2Dtranspose")
			: Layer<float>(name) {}
	};
} // namespace Ritsu