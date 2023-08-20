#pragma once
#include "Activaction.h"
#include "Tensor.h"
#include <cmath>

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class Tahn : public Activaction {
	  public:
		Tahn(const std::string &name = "tahn") : Activaction(name) {}

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

		Tensor compute_derivative(const Tensor &tensor) override {
			Tensor output = tensor;
			computeDerivative(output);
			return output;
		}
		Tensor &compute_derivative(Tensor &tensor) const override {
			computeDerivative(tensor);
			return tensor;
		}

		std::vector<Layer<DType> *> getInputs() const override { return {input}; }
		std::vector<Layer<DType> *> getOutputs() const override { return outputs; }

	  private:
		inline static constexpr DType computeTanh(DType value) {

			const DType e_ = std::exp(-value);
			const DType _e_ = std::exp(value);

			return (e_ - _e_) / (e_ + _e_);
		}

		inline static constexpr DType computeTanhDerivate(DType value) {
			return std::exp(-value) / std::pow(((std::exp(-value) + 1)), 2);
		}

		static void computeDerivative(Tensor &output) {
			/*Iterate through each all elements.    */
			const size_t nrElements = output.getNrElements();

#pragma omp parallel shared(output)
			for (size_t i = 0; i < nrElements; i++) {
				output.getValue<DType>(i) = Tahn::computeTanh(output.getValue<DType>(i));
			}
		}

		void computeActivation(Tensor &tensor) {
			/*Iterate through each all elements.    */
			const size_t nrElements = tensor.getNrElements();

#pragma omp parallel shared(tensor)
			for (size_t i = 0; i < nrElements; i++) {
				tensor.getValue<DType>(i) = Tahn::computeTanh(tensor.getValue<DType>(i));
			}
		}

	  private:
		Layer<DType> *input;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu