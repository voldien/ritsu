#pragma once
#include "Layer.h"

namespace Ritsu {
	/**
	 * @brief 
	 * 
	 */
	class AveragePooling2D : public Layer<float> {

	  public:
		AveragePooling2D(int stride, const std::string &name = "averagepooling") : Layer(name) {}

	  private:
		int stride;
	};
} // namespace Ritsu