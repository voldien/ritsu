#pragma once
#include "Layer.h"

namespace Ritsu {

	class Regularization : public Layer<float> {

	  public:
		Regularization(float L1, float L2, const std::string &name = "Regularization") : Layer<float>(name) {}

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

		Tensor compute_deriviate(const Tensor &tensor) override { return tensor; }
		Tensor &compute_deriviate(Tensor &tensor) const override { return tensor; }

	  private:
		static inline void computeElementSum(Tensor &inputA, const Tensor &inputB) { inputA = inputA + inputB; }

	  private:
		std::vector<Layer<DType> *> *input;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu