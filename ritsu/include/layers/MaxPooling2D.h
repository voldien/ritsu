#pragma once
#include "Layer.h"
#include "Tensor.h"
#include <algorithm>

namespace Ritsu {

	class MaxPooling2D : public Layer<float> {

	  public:
		MaxPooling2D(const int stride[2], const std::string &name = "maxpooling") : Layer(name) {}

		Tensor operator<<(const Tensor &tensor) override { return tensor; }

		Tensor &operator<<(Tensor &tensor) override { return tensor; }

		Tensor operator>>(Tensor &tensor) override { return tensor; }

		Tensor &operator()(Tensor &tensor) override { return tensor; }

	  private:
		void computeMaxPooling2D(const Tensor &tensor, const Tensor &output) {

			const size_t width = 0;
			const size_t height = 0;

			// Verify shape

			for (size_t y = 0; y < height; y++) {
				for (size_t x = 0; x < width; x++) {

					DType maxValue = static_cast<DType>(-999999999);
					for (size_t Sy = 0; Sy < this->stride[0]; Sy++) {
						for (size_t Sx = 0; Sx < this->stride[1]; Sx++) {
						}
					}
				}
			}
		}

		int stride[2];
	};
} // namespace Ritsu