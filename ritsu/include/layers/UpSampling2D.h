#pragma once
#include "Layer.h"
#include <cstdint>

namespace Ritsu {

	/**
	 * @brief
	 *
	 * @tparam T
	 */
	template <typename T> class UpSampling2D : public Layer<T> {
	  public:
		enum class Interpolation { NEAREST, BILINEAR, BICUBIC };

	  public:
		UpSampling2D(uint32_t scale, Interpolation interpolation = Interpolation::NEAREST,
					 const std::string &name = "upscale")
			: Layer<T>(name) {
			this->scale = scale;
		}
		Tensor operator<<(const Tensor &tensor) override {

			// Tensor output({this->units, 1}, DTypeSize);
			//
			// this->compute(tensor, output);

			// return output;
			return tensor;
		}

		Tensor operator>>(Tensor &tensor) override {
			this->computeConv2D(tensor, tensor);
			return tensor;
		}

		Tensor &operator()(Tensor &tensor) override {
			this->computeConv2D(tensor, tensor);
			return tensor;
		}
		template <class U> auto &operator()(U &layer) {

			this->setInputs({&layer});
			layer.setOutputs({this});

			this->build(layer.getShape());

			return *this;
		}

		void setOutputs(const std::vector<Layer<T> *> &layers) override {
			/*	Set input layer */
			this->outputs = layers;
		}

		void setInputs(const std::vector<Layer<T> *> &layers) override {}

		Tensor compute_derivative(const Tensor &tensor) override {
			Tensor output = tensor;

			return output;
		}
		Tensor &compute_derivative(Tensor &tensor) const override { return tensor; }

		std::vector<Layer<T> *> getInputs() const override { return {}; }
		std::vector<Layer<T> *> getOutputs() const override { return {}; }

	  private:
		void computeUpSampling(const Tensor &a, const Tensor &b, Tensor &output) {}

		std::vector<Layer<T> *> inputs;
		std::vector<Layer<T> *> outputs;
		uint32_t scale;
	};
} // namespace Ritsu