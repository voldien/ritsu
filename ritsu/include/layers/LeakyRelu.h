#pragma once
#include "Layer.h"
#include <ctime>
#include <random>

namespace Ritsu {

	class LeakyRelu : public Layer<float> {
	  public:
		LeakyRelu(const DType alpha, const std::string &name = "leaky-relu") : Layer<DType>(name), alpha(alpha) {}

		Tensor operator<<(const Tensor &tensor) override {
			Tensor output = tensor; // Copy
			this->computeReluLeakyActivation(output);
			return output;
		}

		Tensor &operator<<(Tensor &tensor) override {
			this->computeReluLeakyActivation(tensor);
			return tensor;
		}

		Tensor operator>>(Tensor &tensor) override {
			this->computeReluLeakyActivation(tensor);
			return tensor;
		}

		Tensor &operator()(Tensor &tensor) override {
			this->computeReluLeakyActivation(tensor);
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

		Tensor compute_derivative(const Tensor &tensor) override {
			Tensor output = tensor;
			this->computeReluLeakyDerivative(output);
			return output;
		}

		Tensor &compute_derivative(Tensor &tensor) const override {
			this->computeReluLeakyDerivative(tensor);
			return tensor;
		}

	  protected:
#pragma omp declare simd uniform(value, alpha)
		inline static constexpr DType leakyRelu(DType value, const DType alpha) {
			if (value < 0) {
				return value * alpha;
			}
			return std::max<DType>(0, value);
		}
#pragma omp declare simd uniform(value, alpha)
		inline static constexpr DType leakyReluDerivative(DType value, const DType alpha) {
			if (value >= 0) {
				return 0;
			}
			return alpha;
		}

		void computeReluLeakyActivation(Tensor &tensor) const {
			/*Iterate through each all elements.    */
			const size_t nrElements = tensor.getNrElements();
#pragma omp parallel shared(tensor)
			for (size_t i = 0; i < nrElements; i++) {
				tensor.getValue<DType>(i) = LeakyRelu::leakyRelu(tensor.getValue<DType>(i), this->alpha);
			}
		}

		void computeReluLeakyDerivative(Tensor &tensor) const {
			const size_t nrElements = tensor.getNrElements();
#pragma omp parallel shared(tensor)
			for (size_t i = 0; i < nrElements; i++) {
				tensor.getValue<DType>(i) = LeakyRelu::leakyReluDerivative(tensor.getValue<DType>(i), this->alpha);
			}
		}

	  private:
		DType alpha;
		Layer<DType> *input;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu