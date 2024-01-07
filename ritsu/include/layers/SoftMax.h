#pragma once
#include "Activaction.h"
#include <cmath>

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
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

		template <class U> auto &operator()(U &layer) {

			this->setInputs({&layer});
			layer.setOutputs({this});

			this->build(layer.getShape());

			return *this;
		}

		// virtual const Tensor &operator()(const Tensor &tensor) override {
		//	//compute(tensor);
		//	return tensor;
		//}

		void setInputs(const std::vector<Layer<DType> *> &layers) override {}
		void setOutputs(const std::vector<Layer<DType> *> &layers) override {}

		Tensor compute_derivative(const Tensor &tensorLoss) override {}
		Tensor &compute_derivative(Tensor &tensorLoss) const override {}

	  private:
		void computeSoftMax(Tensor &tensor) {
			/*	Iterate through each all elements.    */
			DType Inversesum = 0;
			const size_t nrElements = tensor.getNrElements();

#pragma omp parallel
			for (size_t i = 0; i < nrElements; i++) {
				Inversesum += static_cast<DType>(std::exp(tensor.getValue<DType>(i)));
			}
			Inversesum = 1.0f / Inversesum;
#pragma omp parallel
			for (size_t i = 0; i < nrElements; i++) {
				tensor.getValue<DType>(i) = tensor.getValue<DType>(i) * Inversesum;
			}
		}
	};
} // namespace Ritsu