#pragma once
#include "Layer.h"
#include <array>
#include <cassert>

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class AveragePooling2D : public Layer<float> {

	  public:
		AveragePooling2D(const std::array<uint32_t, 2> &size, const std::string &name = "averagepooling") : Layer(name) {
			this->size = size;
		}

		void build(const Shape<IndexType> &shape) override {

			this->shape = shape;

			this->shape[-2] /= size[0];
			this->shape[-3] /= size[1];

			/*	*/
			assert(this->getShape().getNrDimensions() == 3);
		}

		template <class U> auto &operator()(U &layer) {

			this->setInputs({&layer});
			layer.setOutputs({this});

			this->build(layer.getShape());

			return *this;
		}

		void setOutputs(const std::vector<Layer<float> *> &layers) override {
			/*	Set input layer */
			this->outputs = layers;
		}

		void setInputs(const std::vector<Layer<float> *> &layers) override { /*	*/
		}

		Tensor compute_derivative(const Tensor &tensorLoss) override {}
		Tensor &compute_derivative(Tensor &tensorLoss) const override {}

	  private:
		std::array<uint32_t, 2> size;

		std::vector<Layer<float> *> inputs;
		std::vector<Layer<float> *> outputs;
		uint32_t scale;
	};
} // namespace Ritsu