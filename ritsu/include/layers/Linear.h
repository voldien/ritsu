#pragma once
#include "Activaction.h"
#include <cmath>

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class Linear : public Activaction {
	  public:
		Linear(const DType linear, const std::string &name = "linear") : Activaction(name), linear(linear) {}

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

		void setInputs(const std::vector<Layer<DType> *> &layers) override { this->input = layers[0]; }

		Tensor compute_derivative(const Tensor &tensor) override { return tensor; }
		Tensor &compute_derivative(Tensor &tensor) const override { return tensor; }

		std::vector<Layer<DType> *> getInputs() const override { return {input}; }
		std::vector<Layer<DType> *> getOutputs() const override { return outputs; }

	  private:
		inline static constexpr DType computeLinear(const DType coeff, const DType value) { return coeff * value; }

		inline static constexpr DType computeLinearDerivative(const DType coff) { return coff; }

		void computeActivation(Tensor &tensor) {
			/*Iterate through each all elements.    */
			const size_t nrElements = tensor.getNrElements();

#pragma omp parallel shared(tensor)
			for (size_t i = 0; i < nrElements; i++) {
				tensor.getValue<DType>(i) = Linear::computeLinear(this->linear, tensor.getValue<DType>(i));
			}
		}

		void computeActivationDerivative(Tensor &tensor) {
			/*Iterate through each all elements.    */
			const size_t nrElements = tensor.getNrElements();

#pragma omp parallel shared(tensor)
			for (size_t i = 0; i < nrElements; i++) {
				tensor.getValue<DType>(i) = Linear::computeLinearDerivative(this->linear);
			}
		}

	  private:
		Layer<DType> *input;
		std::vector<Layer<DType> *> outputs;
		DType linear;
	};
} // namespace Ritsu