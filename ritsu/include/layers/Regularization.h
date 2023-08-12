#pragma once
#include "Layer.h"
#include "Tensor.h"

namespace Ritsu {

	class Regularization : public Layer<float> {

	  public:
		Regularization(DType l1 = 0.0f, DType L2 = 0.0f, const std::string &name = "Regularization")
			: Layer<float>(name), l1(l1), l2(l2) {}

		Tensor operator<<(const Tensor &tensor) override {

			Tensor output = tensor;
			// this->computeElementSum(this-> output);
			return output;
		}

		Tensor &operator<<(Tensor &tensor) override {
			//	this->computeElementSum(tensor);
			return tensor;
		}

		Tensor operator>>(Tensor &tensor) override {
			// this->computeElementSum(tensor);
			return tensor;
		}

		Tensor &operator()(Tensor &tensor) override {
			// this->computeElementSum(tensor);
			return tensor;
		}

		template <class U> auto &operator()(std::vector<U> &layers) {
			// this->setInputs({&layer});
			// layer.setOutputs({this});
			// return *this;
		}

		void setInputs(const std::vector<Layer<DType> *> &layers) override {}
		void setOutputs(const std::vector<Layer<DType> *> &layers) override {}

		Tensor compute_derivative(const Tensor &tensor) override { return tensor; }
		Tensor &compute_derivative(Tensor &tensor) const override { return tensor; }

	  private:
		static inline void computeElementSum(Tensor &inputA, const Tensor &inputB) { inputA = inputA + inputB; }

		static void computeL1(const Tensor &x0, const Tensor &x1, Tensor &output) {
			//#pragma omp parallel shared(tensor)
			//			for (size_t i = 0; i < tensor.getNrElements(); i++) {
			//				tensor.getValue<DType>(i) = computeSigmoidDerivate(tensor.getValue<DType>(i));
			//			}
		}

		static void computeL2(const Tensor &x0, const Tensor &x1, Tensor &output) {
			//#pragma omp parallel shared(tensor)
			//			for (size_t i = 0; i < tensor.getNrElements(); i++) {
			//				tensor.getValue<DType>(i) = computeSigmoidDerivate(tensor.getValue<DType>(i));
			//			}
		}

	  private:
		DType l1;
		DType l2;
		std::vector<Layer<DType> *> *input;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu