#pragma once
#include "Activaction.h"
#include "Activations.h"
#include <cmath>

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class Sigmoid : public Activaction {
	  public:
		Sigmoid(const std::string &name = "sigmoid") : Activaction(name) {}

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

		void setInputs(const std::vector<Layer<DType> *> &layers) override {
			this->input = layers[0];

			/*	*/
			this->build(this->input->getShape());
		}

		void build(const Shape<IndexType> &shape) override { this->shape = shape; }

		std::vector<Layer<DType> *> getInputs() const override { return {input}; }
		std::vector<Layer<DType> *> getOutputs() const override { return outputs; }

		Tensor<float> compute_derivative(const Tensor<float> &tensor) override {
			Tensor<float> result(tensor.getShape());
			this->computeSigmoidDerivative(result);
			return result;
		}

		Tensor<float> &compute_derivative(Tensor<float> &tensor) const override {
			this->computeSigmoidDerivative(tensor);
			return tensor;
		}

	  private:
		void computeSigmoidDerivative(Tensor<float> &tensor) const noexcept {
			const size_t nrElements = tensor.getNrElements();
#pragma omp parallel shared(tensor)
			for (size_t i = 0; i < tensor.getNrElements(); i++) {
				tensor.getValue<DType>(i) = computeSigmoidDerivate(tensor.getValue<DType>(i));
			}
		}

		void computeActivation(Tensor<float> &tensor) noexcept {
			/*Iterate through each all elements.    */
			const size_t nrElements = tensor.getNrElements();

#pragma omp parallel shared(tensor)
			for (size_t i = 0; i < nrElements; i++) {
				tensor.getValue<DType>(i) = computeSigmoid(tensor.getValue<DType>(i));
			}
		}

	  private:
		Layer<DType> *input;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu