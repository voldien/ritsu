#pragma once
#include "Layer.h"

namespace Ritsu {

	class Add : public Layer<float> {

	  public:
		Add(const Layer<float> &a, const Layer<float> b, const std::string &name = "Add") : Layer<float>(name) {}

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

	  private:
		static inline void computeElementSum(Tensor &inputA, const Tensor &inputB) {
			const size_t nrElements = inputA.getNrElements();
			inputA = inputA + inputB;
		}

	  private:
		std::vector<Layer<DType> *> *input;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu