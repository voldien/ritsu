#pragma once
#include "Activaction.h"
#include "layers/Layer.h"
#include <cmath>

namespace Ritsu {

	class Relu : public Activaction {
	  public:
		Relu(const std::string &name = "relu") : Activaction(name) {}

		Tensor operator<<(const Tensor &tensor) override {
			Tensor output = tensor; // Copy
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
			this->shape = layers[0]->getNrDimension();
			this->input = layers[0];
		}

		std::vector<Layer<DType> *> getInputs() const override { return {input}; }
		std::vector<Layer<DType> *> getOutputs() const override { return outputs; }

	  protected:
#pragma omp declare simd uniform(x)
		DType relu(DType x) { return std::max<float>(0, x); }

		void computeActivation(Tensor &tensor) {
			/*Iterate through each all elements.    */
			const size_t nrElements = tensor.getNrElements();
			#pragma omp parallel
			for (size_t i = 0; i < nrElements; i++) {
				tensor.getValue<float>(i) = this->relu(tensor.getValue<float>(i));
			}
		}

	  private:
		Layer<DType> *input;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu