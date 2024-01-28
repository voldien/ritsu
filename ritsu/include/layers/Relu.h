#pragma once
#include "Activaction.h"
#include "Tensor.h"
#include "layers/Layer.h"
#include <cmath>

namespace Ritsu {
	
	/**
	 * @brief 
	 * 
	 */
	class Relu : public Activaction {
	  public:
		Relu(const std::string &name = "relu") : Activaction(name) {}

		Tensor<float> operator<<(const Tensor<float> &tensor) override {
			Tensor<float> output = tensor; // Copy
			this->computeReluActivation(output);
			return output;
		}

		Tensor<float> &operator<<(Tensor<float> &tensor) override {
			this->computeReluActivation(tensor);
			return tensor;
		}

		Tensor<float> operator>>(Tensor<float> &tensor) override {
			this->computeReluActivation(tensor);
			return tensor;
		}

		Tensor<float> &operator()(Tensor<float> &tensor) override {
			this->computeReluActivation(tensor);
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
			this->shape = layers[0]->getShape();
			this->input = layers[0];
		}

		std::vector<Layer<DType> *> getInputs() const override { return {input}; }
		std::vector<Layer<DType> *> getOutputs() const override { return outputs; }

		Tensor<float> compute_derivative(const Tensor<float> &tensorLoss) override { return tensorLoss; }
		Tensor<float> &compute_derivative(Tensor<float> &tensorLoss) const override {
			computeDeriviate(tensorLoss);
			return tensorLoss;
		}

	  protected:

		void computeReluActivation(Tensor<float> &tensor) {
			/*Iterate through each all elements.    */
			const size_t nrElements = tensor.getNrElements();
#pragma omp parallel shared(tensor)
			for (size_t i = 0; i < nrElements; i++) {
				tensor.getValue<DType>(i) = relu(tensor.getValue<DType>(i));
			}
		}

		static void computeDeriviate(Tensor<float> &tensor) {
			const size_t nrElements = tensor.getNrElements();
#pragma omp parallel shared(tensor)
			for (size_t i = 0; i < nrElements; i++) {
				tensor.getValue<DType>(i) = reluDeriviate(tensor.getValue<DType>(i));
			}
		}

	  private:
		Layer<DType> *input;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu