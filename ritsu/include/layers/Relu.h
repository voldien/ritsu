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

		Tensor compute_deriviate(const Tensor &tensor) override { return tensor; }
		Tensor &compute_deriviate(Tensor &tensor) const override {
			computeDeriviate(tensor);
			return tensor;
		}

	  protected:
#pragma omp declare simd uniform(value)
		DType relu(DType value) { return std::max<float>(0, value); }

		void computeReluActivation(Tensor &tensor) {
			/*Iterate through each all elements.    */
			const size_t nrElements = tensor.getNrElements();
#pragma omp parallel
			for (size_t i = 0; i < nrElements; i++) {
				tensor.getValue<float>(i) = this->relu(tensor.getValue<float>(i));
			}
		}

		static void computeDeriviate(Tensor &tensor) {}

	  private:
		Layer<DType> *input;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu