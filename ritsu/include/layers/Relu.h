#pragma once
#include "Activaction.h"
#include "Tensor.h"
#include "layers/Layer.h"
#include <cmath>

namespace Ritsu {

	class Relu : public Activaction {
	  public:
		Relu(const std::string &name = "relu") : Activaction(name) {}

		Tensor operator<<(const Tensor &tensor) override {
			Tensor output = tensor; // Copy
			this->computeReluActivation(output);
			return output;
		}

		Tensor &operator<<(Tensor &tensor) override {
			this->computeReluActivation(tensor);
			return tensor;
		}

		Tensor operator>>(Tensor &tensor) override {
			this->computeReluActivation(tensor);
			return tensor;
		}

		Tensor &operator()(Tensor &tensor) override {
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

		Tensor compute_derivative(const Tensor &tensor) override { return tensor; }
		Tensor &compute_derivative(Tensor &tensor) const override {
			computeDeriviate(tensor);
			return tensor;
		}

	  protected:
#pragma omp declare simd uniform(value)
		inline static constexpr DType relu(DType value) { return std::max<DType>(0, value); }

#pragma omp declare simd uniform(value)
		inline static constexpr DType reluDeriviate(DType value) {
			if (value >= 0) {
				return 1;
			}
			return 0;
		}

		void computeReluActivation(Tensor &tensor) {
			/*Iterate through each all elements.    */
			const size_t nrElements = tensor.getNrElements();
#pragma omp parallel shared(tensor)
			for (size_t i = 0; i < nrElements; i++) {
				tensor.getValue<DType>(i) = Relu::relu(tensor.getValue<DType>(i));
			}
		}

		static void computeDeriviate(Tensor &tensor) {
			const size_t nrElements = tensor.getNrElements();
#pragma omp parallel shared(tensor)
			for (size_t i = 0; i < nrElements; i++) {
				tensor.getValue<DType>(i) = Relu::reluDeriviate(tensor.getValue<DType>(i));
			}
		}

	  private:
		Layer<DType> *input;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu