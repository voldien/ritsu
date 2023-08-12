#pragma once
#include "Layer.h"

namespace Ritsu {
	class AveragePooling : public Layer<float> {

	  public:
		AveragePooling(int stride, const std::string &name = "maxpooling") : Layer(name) {}

	  private:
		int stride;
	};
}