#pragma once
#include "Add.h"
#include "Layer.h"

namespace Ritsu {

	class Divide : public Add {
		//  kernel_initializer='glorot_uniform',
		//    bias_initializer='zeros',

	  public:
		Divide(Layer<float> &a, Layer<float> b) {}

		virtual void computeActivation(Tensor &tensor) {
			/*Iterate through each all elements.    */
			size_t nrElements = tensor.getNrElements();

#pragma omp parallel shared(tensor)
			for (size_t i = 0; i < nrElements; i++) {
				tensor.getValue<float>(i) = this->computeSigmoid(tensor.getValue<float>(i));
			}
		}

	  private:
	};
} // namespace Ritsu