#pragma once
#include "Layer.h"
#include "Tensor.h"
#include <algorithm>

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class MaxPooling2D : public Layer<float> {

	  public:
		MaxPooling2D(const std::array<uint32_t, 2> &size, const std::string &name = "maxpooling") : Layer(name) {
			this->size = size;
		}

		Tensor<float> operator<<(const Tensor<float> &tensor) override { return tensor; }

		Tensor<float> &operator<<(Tensor<float> &tensor) override { return tensor; }

		Tensor<float> operator>>(Tensor<float> &tensor) override { return tensor; }

		Tensor<float> &operator()(Tensor<float> &tensor) override { return tensor; }

		template <class U> auto &operator()(U &layer) {

			this->setInputs({&layer});
			layer.setOutputs({this});

			this->build(layer.getShape());

			return *this;
		}

		void setInputs(const std::vector<Layer<DType> *> &layers) override {}
		void setOutputs(const std::vector<Layer<DType> *> &layers) override {}

		Tensor<float> compute_derivative(const Tensor<float> &tensor) override { return tensor; }
		Tensor<float> &compute_derivative(Tensor<float> &tensor) const override { return tensor; }

	  private:
		void computeMaxPooling2D(const Tensor<float> &tensor, const Tensor<float> &output) {

			const size_t width = 0;
			const size_t height = 0;

			// Verify shape

			for (size_t y = 0; y < height; y++) {
				for (size_t x = 0; x < width; x++) {

					DType maxValue = static_cast<DType>(-999999999);
					for (size_t Sy = 0; Sy < this->size[0]; Sy++) {
						for (size_t Sx = 0; Sx < this->size[1]; Sx++) {
							/*	*/
						}
					}
				}
			}
		}

		std::array<uint32_t, 2> size;
	};
} // namespace Ritsu