#pragma once
#include "Layer.h"
#include "Tensor.h"

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

		Tensor<float> operator<<(const Tensor<float> &tensor) override {

			// Tensor<float> output({this->units, 1}, DTypeSize);
			//
			// this->compute(tensor, output);

			// return output;
			return tensor;
		}

		Tensor<float> operator>>(Tensor<float> &tensor) override {
			this->computeUpSampling(tensor, tensor);
			return tensor;
		}

		Tensor<float> &operator()(Tensor<float> &tensor) override {
			this->computeUpSampling(tensor, tensor);
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

		void build(const Shape<typename Layer<T>::IndexType> &shape) override {

			this->shape = shape;
			this->shape[0] /= 2;
			this->shape[1] /= 2;

			/*	*/
			assert(this->getShape().getNrDimensions() == 3);
		}

		Tensor<float> compute_derivative(const Tensor<float> &tensor) override {
			Tensor<float> output = tensor;

			return output;
		}
		Tensor<float> &compute_derivative(Tensor<float> &tensor) const override { return tensor; }

		std::vector<Layer<T> *> getInputs() const override { return {}; }
		std::vector<Layer<T> *> getOutputs() const override { return {}; }

	  private:
		void computeUpSampling(const Tensor<float> &a, const Tensor<float> &b) { //, Tensor<float> &output) {}
			switch (this->interpolation) {
			case Interpolation::NEAREST:
				break;
			case Interpolation::BILINEAR:
				break;
			case Interpolation::BICUBIC:
				break;
			default:
				break;
			}
		}

		inline static void computeUpscaleNearest() {}
		inline static void computeUpscaleBilinear() {}
		inline static void computeUpscaleBiCubic() {}

		std::vector<Layer<T> *> inputs;
		std::vector<Layer<T> *> outputs;
		uint32_t scale;
		Interpolation interpolation;
	};
} // namespace Ritsu