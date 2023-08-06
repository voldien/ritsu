#pragma once
#include "../Math.h"
#include "Activaction.h"
#include <cmath>

namespace Ritsu {

	class SoftMax : public Activaction {
	  public:
		SoftMax() {}
		~SoftMax() override {}

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

		Tensor &operator()(Tensor &tensor) override {
			compute(tensor);
			return tensor;
		}

		// virtual const Tensor &operator()(const Tensor &tensor) override {
		//	//compute(tensor);
		//	return tensor;
		//}

	  private:
		void compute(Tensor &tensor) {
			/*Iterate through each all elements.    */
			float sum = 0;
			const size_t nrElements = tensor.getNrElements();
			for (size_t i = 0; i < nrElements; i++) {
				sum += std::pow(static_cast<DType>(Math::E), tensor.getValue<float>(i));
			}

			for (size_t i = 0; i < nrElements; i++) {
				tensor.getValue<float>(i) = tensor.getValue<float>(i) / sum;
			}
		}
	};
} // namespace Ritsu