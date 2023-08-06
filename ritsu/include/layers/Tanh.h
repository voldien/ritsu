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

		Tensor compute_deriviate(const Tensor &tensor) override { return tensor; }
		Tensor &compute_deriviate(Tensor &tensor) const override { return tensor; }

		std::vector<Layer<DType> *> getInputs() const override { return {input}; }
		std::vector<Layer<DType> *> getOutputs() const override { return outputs; }

	  private:
		inline float computeTanh(DType x) const {
			
			const float e_ = std::pow(static_cast<DType>(Math::E), -x);
			const float _e_ = std::pow(static_cast<DType>(Math::E), x);

			return (e_ - _e_) / (e_ + _e_);
		}
		inline float computeTanhDerivate(float value) const { return std::exp(-value) / std::pow(((std::exp(-value) + 1)), 2); }

		void computeActivation(Tensor &tensor) {
			/*Iterate through each all elements.    */
			size_t nrElements = tensor.getNrElements();

#pragma omp parallel shared(tensor)
			for (size_t i = 0; i < nrElements; i++) {
				tensor.getValue<float>(i) = this->computeTanh(tensor.getValue<float>(i));
			}
		}

	  private:
		Layer<DType> *input;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu