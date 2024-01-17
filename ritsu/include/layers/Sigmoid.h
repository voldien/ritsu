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

		Tensor operator<<(const Tensor &tensor) override {

			Tensor output = tensor;

			this->computeActivation(output);
			return output;
		}

		Tensor &operator<<(Tensor &tensor) override {
			this->computeActivation(tensor);
			return tensor;
		}

		Tensor operator>>(Tensor &tensor) override {
			this->computeActivation(tensor);
			return tensor;
		}

		Tensor &operator()(Tensor &tensor) override {
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

		Tensor compute_derivative(const Tensor &tensor) override {
			Tensor result(tensor.getShape());
			this->computeSigmoidDerivative(result);
			return result;
		}

		Tensor &compute_derivative(Tensor &tensor) const override {
			this->computeSigmoidDerivative(tensor);
			return tensor;
		}

	  private:

		void computeSigmoidDerivative(Tensor &tensor) const noexcept {
#pragma omp parallel shared(tensor)
			for (size_t i = 0; i < tensor.getNrElements(); i++) {
				tensor.getValue<DType>(i) = computeSigmoidDerivate(tensor.getValue<DType>(i));
			}
		}

		void computeActivation(Tensor &tensor) noexcept {
			/*Iterate through each all elements.    */
			size_t nrElements = tensor.getNrElements();

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