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
		UpSampling2D(uint32_t scale, const std::string &name = "upscale") : Layer<T>(name) {}

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
		// void compute(Tensor &a, Tensor &b) { return a + b; }

		std::vector<Layer<T> *> inputs;
		std::vector<Layer<T> *> outputs;
	};
} // namespace Ritsu