#pragma once
#include "Layer.h"
#include <cstdarg>

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

		template <class U> Add &operator()(U &layer...) {
			return *this;
			va_list args;
			va_start(args, layer);

			Layer<DType> *refB = nullptr;
			for (refB = &layer; refB != nullptr; refB = va_arg(args, Layer<DType> *)) {

				this->inputs.push_back(refB);
				// this->setInputs({&layer});
				refB->setOutputs({this});
			}

			va_end(args);

			this->build(layer.getShape());

			return *this;
		}

		void build(const Shape<IndexType> &shape) override { this->shape = shape; }

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