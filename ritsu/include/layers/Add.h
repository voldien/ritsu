#pragma once
#include "Layer.h"

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class Add : public Layer<float> {

	  public:
		Add(const std::string &name = "Add") : Layer<float>(name) {}

		Tensor operator<<(const Tensor &tensor) override {

			Tensor output = tensor;
			// this->computeElementSum(this-> output);
			return output;
		}

		Tensor &operator<<(Tensor &tensor) override {
			//	this->computeElementSum(tensor);
			return tensor;
		}

		Tensor operator>>(Tensor &tensor) override { return tensor; }

		Tensor &operator()(Tensor &tensor) override {
			// this->computeElementSum(tensor);
			return tensor;
		}

		template <class U> auto &operator()(std::vector<U> &layers) {
			// this->setInputs({&layer});
			// layer.setOutputs({this});
			// return *this;
		}

		void setInputs(const std::vector<Layer<DType> *> &layers) override { this->inputs = layers; }
		void setOutputs(const std::vector<Layer<DType> *> &layers) override { this->outputs = layers; }

		Tensor compute_derivative(const Tensor &tensor) override { return tensor; }
		Tensor &compute_derivative(Tensor &tensor) const override { return tensor; }

	  private:
		static inline void computeElementSum(Tensor &inputA, const Tensor &inputB) { inputA = inputA + inputB; }

	  private:
		std::vector<Layer<DType> *> inputs;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu