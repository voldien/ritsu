#pragma once
#include "Activaction.h"
#include <cmath>

namespace Ritsu {

	class SoftMax : public Activaction {
	  public:
		SoftMax() : Activaction() {}
		virtual ~SoftMax() {}

		 Tensor operator<<(const Tensor &tensor) override {
			// compute(tensor);
			return tensor;
		}

		 Tensor operator>>(Tensor &tensor) override {
			compute(tensor);
			return tensor;
		}

		// virtual Tensor operator()(Tensor &tensor) {
		//	compute(tensor);
		//	return tensor;
		//}

		virtual Tensor &operator()(Tensor &tensor) override {
			compute(tensor);
			return tensor;
		}

		// virtual const Tensor &operator()(const Tensor &tensor) override {
		//	//compute(tensor);
		//	return tensor;
		//}

	  private:
		void compute(Tensor &X) {
			/*Iterate through each all elements.    */
			const size_t nrElements = X.getNrElements();
			for (size_t i = 0; i < nrElements; i++) {
				X.getValue<float>(i) = std::max<float>(0, X.getValue<float>(i));
			}
		}
	};
} // namespace Ritsu