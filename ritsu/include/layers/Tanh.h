#pragma once
#include "Activaction.h"
#include <cmath>

namespace Ritsu {

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

		std::vector<Layer<DType> *> getInputs() const override { return {input}; }
		std::vector<Layer<DType> *> getOutputs() const override { return outputs; }

	  private:
		inline float computeSigmoid(float x) const { return 1.0f / (1.0 + std::pow(2.718281828459045, -x)); }
		inline float computeSigmoidDerivate(float x) const { return std::exp(-x) / std::pow(((std::exp(-x) + 1)), 2); }

		void computeActivation(Tensor &X) {
			/*Iterate through each all elements.    */
			size_t nrElements = X.getNrElements();
			
			#pragma omp parallel shared(X)
			for (size_t i = 0; i < nrElements; i++) {
				X.getValue<float>(i) = this->computeSigmoid(X.getValue<float>(i));
			}
		}

	  private:
		Layer<DType> *input;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu