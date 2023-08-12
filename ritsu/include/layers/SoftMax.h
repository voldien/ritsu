#pragma once
#include "../Math.h"
#include "Activaction.h"
#include <cmath>

namespace Ritsu {

	class SoftMax : public Activaction {
	  public:
		SoftMax(const std::string &name = "softmax") : Activaction(name) {}
		~SoftMax() override {}

		Tensor operator<<(const Tensor &tensor) override {
			// compute(tensor);
			return tensor;
		}

		Tensor operator>>(Tensor &tensor) override {
			this->computeSoftMax(tensor);
			return tensor;
		}

		// virtual Tensor operator()(Tensor &tensor) {
		//	compute(tensor);
		//	return tensor;
		//}

		Tensor &operator()(Tensor &tensor) override {
			this->computeSoftMax(tensor);
			return tensor;
		}

		// virtual const Tensor &operator()(const Tensor &tensor) override {
		//	//compute(tensor);
		//	return tensor;
		//}

	  private:
		void computeSoftMax(Tensor &tensor) {
			/*Iterate through each all elements.    */
			DType sum = 0;
			const size_t nrElements = tensor.getNrElements();
#pragma omp parallel
			for (size_t i = 0; i < nrElements; i++) {
				sum += static_cast<DType>(std::exp(tensor.getValue<DType>(i)));
			}
#pragma omp parallel
			for (size_t i = 0; i < nrElements; i++) {
				tensor.getValue<DType>(i) = tensor.getValue<DType>(i) / sum;
			}
		}
		
	};
} // namespace Ritsu