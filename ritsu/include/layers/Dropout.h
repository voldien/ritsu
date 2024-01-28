#pragma once
#include "Layer.h"
#include "Random.h"

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class Dropout : public Layer<float> {

	  public:
		Dropout(const DType perc, const size_t seed = 0, const std::string &name = "dropout")
			: Layer(name), perc(perc) {
			this->random = new RandomBernoulli<DType>(perc);
		}

		Tensor<float> &operator<<(Tensor<float> &tensor) override {
			this->computeDropout(tensor);
			return tensor;
		}

		Tensor<float> operator<<(const Tensor<float> &tensor) override {
			Tensor<float> tmpOutput = tensor;
			this->computeDropout(tmpOutput);
			return tmpOutput;
		}

		Tensor<float> &operator()(Tensor<float> &tensor) override {
			this->computeDropout(tensor);
			return tensor;
		}

		template <class U> auto &operator()(U &layer) {

			this->setInputs({&layer});
			layer.setOutputs({this});

			return *this;
		}

		void setInputs(const std::vector<Layer<DType> *> &layers) override {
			this->input = layers[0];

			this->shape = this->input->getShape();
		}

		void setOutputs(const std::vector<Layer<DType> *> &layers) override {
			/*	Set input layer */
			this->outputs = layers;
		}

		std::vector<Layer<DType> *> getInputs() const override { return {input}; }
		std::vector<Layer<DType> *> getOutputs() const override { return outputs; }

		void build(const Shape<IndexType> &shape) override {}

		Tensor<float> compute_derivative(const Tensor<float> &tensorLoss) override { return tensorLoss; }
		Tensor<float> &compute_derivative(Tensor<float> &tensorLoss) const override { return tensorLoss; }

	  private:
		void computeDropout(Tensor<float> &tensor) { /*	Iterate through each all elements.    */

			/*	*/
		}

		/*	*/
		DType perc;
		Random<DType> *random;
		Layer<DType> *input;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu