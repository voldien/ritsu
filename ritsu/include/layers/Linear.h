#pragma once
#include "Activaction.h"
#include "Activations.h"
#include <cmath>

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class Linear : public Activaction {
	  public:
		Linear(const DType linear, const std::string &name = "linear") : Activaction(name), linear(linear) {}

		Tensor<float> operator<<(const Tensor<float> &tensor) override {

			Tensor<float> output = tensor;
			this->computeActivation(output);
			return output;
		}

		Tensor<float> &operator<<(Tensor<float> &tensor) override {
			this->computeActivation(tensor);
			return tensor;
		}

		Tensor<float> operator>>(Tensor<float> &tensor) override {
			this->computeActivation(tensor);
			return tensor;
		}

		Tensor<float> &operator()(Tensor<float> &tensor) override {
			this->computeActivation(tensor);
			return tensor;
		}

		template <class U> auto &operator()(U &layer) {

			this->setInputs({&layer});
			layer.setOutputs({this});

			return *this;
		}

		void setOutputs(const std::vector<Layer<DType> *> &layers) override {
			/*	Set input layer */
			this->outputs = layers;
		}

		void setInputs(const std::vector<Layer<DType> *> &layers) override { this->input = layers[0]; }

		Tensor<float> compute_derivative(const Tensor<float> &tensor) override { return tensor; }
		Tensor<float> &compute_derivative(Tensor<float> &tensor) const override { return tensor; }

		std::vector<Layer<DType> *> getInputs() const override { return {input}; }
		std::vector<Layer<DType> *> getOutputs() const override { return outputs; }

	  private:
		void computeActivation(Tensor<float> &tensor) {
			/*Iterate through each all elements.    */
			const size_t nrElements = tensor.getNrElements();

#pragma omp parallel for simd shared(tensor)
			for (size_t i = 0; i < nrElements; i++) {
				tensor.getValue<DType>(i) = computeLinear(this->linear, tensor.getValue<DType>(i));
			}
		}

		void computeActivationDerivative(Tensor<float> &tensor) {
			/*Iterate through each all elements.    */
			const size_t nrElements = tensor.getNrElements();

#pragma omp parallel for shared(tensor)
			for (size_t i = 0; i < nrElements; i++) {
				tensor.getValue<DType>(i) = computeLinearDerivative(this->linear);
			}
		}

	  private:
		Layer<DType> *input;
		std::vector<Layer<DType> *> outputs;
		const DType linear;
	};
} // namespace Ritsu