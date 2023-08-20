#pragma once
#include "Layer.h"
#include "Tensor.h"

namespace Ritsu {

	class Regularization : public Layer<float> {

	  public:
		Regularization(const DType L1 = 0, const DType L2 = 0, const std::string &name = "Regularization")
			: Layer<float>(name), l1(L1), l2(L2) {}

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

		template <class U> auto &operator()(U &layer) {

			this->setInputs({&layer});
			layer.setOutputs({this});

			return *this;
		}

		void setInputs(const std::vector<Layer<DType> *> &layers) override {}
		void setOutputs(const std::vector<Layer<DType> *> &layers) override {}

		Tensor compute_derivative(const Tensor &tensorLoss) override {
			Tensor output;
			
			if (this->l1 > 0) {
				computeL1(tensorLoss, this->l1, output);
			}
			if (this->l2 > 0) {
				computeL2(tensorLoss, this->l2, output);
			}

			return output;
		}
		Tensor &compute_derivative(Tensor &tensorLoss) const override {

			Tensor output;
			if (this->l1 > 0) {
				computeL1(tensorLoss, this->l1, output);
			}
			if (this->l2 > 0) {
				computeL2(tensorLoss, this->l2, output);
			}

			return tensorLoss;
		}

	  private:
		static inline void computeElementSum(Tensor &inputA, const Tensor &inputB) { inputA = inputA + inputB; }

		static void computeL1(const Tensor &x0, const DType L1, Tensor &output) {
			//#pragma omp parallel shared(tensor)
			//			for (size_t i = 0; i < tensor.getNrElements(); i++) {
			//				tensor.getValue<DType>(i) = computeSigmoidDerivate(tensor.getValue<DType>(i));
			//			}
		}

		static void computeL2(const Tensor &x0, const DType L2, Tensor &output) {
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