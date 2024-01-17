#pragma once
#include "Layer.h"
#include "Random.h"
#include <ctime>
#include <random>

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class GuassianNoise : public Layer<float> {
	  public:
		GuassianNoise(const DType mean, const DType stddev, const std::string &name = "noise") : Layer(name) {
			this->random = new RandomNormal<DType>(stddev, mean);
		}
		~GuassianNoise() override { delete this->random; }

		Tensor &operator<<(Tensor &tensor) override {
			this->applyNoise(tensor);
			return tensor;
		}

		Tensor operator<<(const Tensor &tensor) override {
			Tensor tmp = tensor;
			this->applyNoise(tmp);
			return tmp;
		}

		Tensor operator>>(Tensor &tensor) override {
			this->applyNoise(tensor);
			return tensor;
		}

		Tensor &operator()(Tensor &tensor) override {
			this->applyNoise(tensor);
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
			this->shape = this->input->getShape();
		}

		std::vector<Layer<DType> *> getInputs() const override { return {input}; }
		std::vector<Layer<DType> *> getOutputs() const override { return outputs; }

		/*	No derivative.	*/
		Tensor compute_derivative(const Tensor &tensor) override { return tensor; }
		Tensor &compute_derivative(Tensor &tensor) const override { return tensor; }

	  private:
		/*	*/
		Layer<DType> *input;
		std::vector<Layer<DType> *> outputs;

	  protected:
		void applyNoise(Tensor &tensor) noexcept {
			/*Iterate through each all elements.    */
			const size_t nrElements = tensor.getNrElements();
#pragma omp parallel for simd shared(tensor)
			for (size_t i = 0; i < nrElements; i++) {
				tensor.getValue<DType>(i) += this->random->rand();
			}
		}

	  private:
		Random<DType> *random;
	};
} // namespace Ritsu