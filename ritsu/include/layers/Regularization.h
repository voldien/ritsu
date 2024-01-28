#pragma once
#include "Layer.h"
#include "Tensor.h"
#include <cassert>
#include <cmath>

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class Regularization : public Layer<float> {

	  public:
		Regularization(const DType L1 = 0, const DType L2 = 0, const std::string &name = "Regularization")
			: Layer<float>(name), l1(L1), l2(L2) {}

		Tensor<float> operator<<(const Tensor<float> &tensor) override {
			Tensor<float> output = std::move(tensor);
			return output;
		}

		Tensor<float> &operator<<(Tensor<float> &tensor) override { return tensor; }

		Tensor<float> operator>>(Tensor<float> &tensor) override { return tensor; }

		Tensor<float> &operator()(Tensor<float> &tensor) override { return tensor; }

		template <class U> auto &operator()(U &layer) {

			this->setInputs({&layer});
			layer.setOutputs({this});

			return *this;
		}

		void setOutputs(const std::vector<Layer<DType> *> &layers) override {

			/*	Set input layer */
			this->outputs = layers;

			// TODO verify flatten

			/*	*/
			this->build(this->getInputs()[0]->getShape());
		}

		void setInputs(const std::vector<Layer<DType> *> &layers) override {

			// TODO verify flatten
			Layer<DType> *inputLayer = layers[0];
			if (inputLayer->getShape().getNrDimensions() == 1) {
			}

			this->input = layers[0];
			this->shape = this->input->getShape();

			assert(layers.size() == 1);
		}

		std::vector<Layer<DType> *> getInputs() const override { return {input}; }
		std::vector<Layer<DType> *> getOutputs() const override { return outputs; }

		Tensor<float> compute_derivative(const Tensor<float> &tensorLoss) override {
			Tensor<float> output(tensorLoss.getShape());

			if (this->l1 > 0) {
				Regularization::computeL1(tensorLoss, this->l1, output);
			}
			if (this->l2 > 0) {
				Regularization::computeL2(tensorLoss, this->l2, output);
			}

			return output;
		}

		Tensor<float> &compute_derivative(Tensor<float> &tensorLoss) const override {

			Tensor<float> output(this->getShape());

			if (this->l1 > 0) {
				computeL1(tensorLoss, this->l1, output);
			}
			if (this->l2 > 0) {
				computeL2(tensorLoss, this->l2, output);
			}

			return tensorLoss;
		}

	  private:
		static inline void computeElementSum(Tensor<float> &inputA, const Tensor<float> &inputB) { inputA = inputA + inputB; }

		// TODO: relocate
		static void computeL1(const Tensor<float> &tensor, const DType L1, Tensor<float> &output) noexcept {
			/*	*/
#pragma omp parallel shared(output, tensor)
			for (size_t i = 0; i < tensor.getNrElements(); i++) {
				output.getValue<DType>(i) = L1 * std::abs(tensor.getValue<DType>(i));
			}
		}

		static void computeL2(const Tensor<float> &tensor, const DType L2, Tensor<float> &output) noexcept {
			/*	*/
#pragma omp parallel shared(output, tensor)
			for (size_t i = 0; i < tensor.getNrElements(); i++) {
				output.getValue<DType>(i) = L2 * (tensor.getValue<DType>(i) * tensor.getValue<DType>(i));
			}
		}

	  private:
		DType l1;
		DType l2;

		Layer<DType> *input;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu