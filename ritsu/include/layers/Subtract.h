#pragma once
#include "Add.h"
#include "Layer.h"

namespace Ritsu {

	class Subtract : public Add {
		//  kernel_initializer='glorot_uniform',
		//    bias_initializer='zeros',

	  public:
		Subtract(Layer<float> &a, Layer<float> b) {}

		void computeActivation(Tensor &X) {
			/*Iterate through each all elements.    */
			size_t nrElements = X.getNrElements();

#pragma omp parallel shared(X)
			for (size_t i = 0; i < nrElements; i++) {
				X.getValue<float>(i) = this->computeSigmoid(X.getValue<float>(i));
			}
		}

	  private:
	};
} // namespace Ritsu