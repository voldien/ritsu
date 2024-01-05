#pragma once
#include "Layer.h"
#include <cassert>

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class AveragePooling2D : public Layer<float> {

	  public:
		AveragePooling2D(int stride, const std::string &name = "averagepooling") : Layer(name) {}

		void build(const Shape<IndexType> &shape) override {

			this->shape = shape;
			this->shape[0] /= this->stride;
			this->shape[1] /= this->stride;

			/*	*/
			assert(this->getShape().getNrDimensions() == 3);
		}

		void setOutputs(const std::vector<Layer<float> *> &layers) override {
			/*	Set input layer */
			this->outputs = layers;
		}

		void setInputs(const std::vector<Layer<float> *> &layers) override {}

	  private:
		int stride;

		std::vector<Layer<float> *> inputs;
		std::vector<Layer<float> *> outputs;
		uint32_t scale;
	};
} // namespace Ritsu