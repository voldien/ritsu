#pragma once
#include "Layer.h"

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class CategoryEncoding : public Layer<float> {
		enum class OutputMode {
			OneHot,
			Count,
		};

	  public:
		CategoryEncoding(const size_t num_tokens, OutputMode outputMode = OutputMode::OneHot,
						 const std::string &name = "CategoryEncoding")
			: Layer<float>(name) {
			this->nrTokens = num_tokens;
			this->mode = outputMode;
		}

		Tensor<float> &operator()(Tensor<float> &tensor) override { return tensor.flatten(); }

		Tensor<float> &operator<<(Tensor<float> &tensor) override { return tensor.flatten(); }

		Tensor<float> operator<<(const Tensor<float> &tensor) override {
			Tensor<float> tmp = tensor;
			return tmp.flatten();
		}

		template <class U> auto &operator()(U &layer) {

			this->setInputs({&layer});
			layer.setOutputs({this});

			this->build(this->getShape());

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

		Tensor<float> compute_derivative(const Tensor<float> &tensor) override { return tensor; }
		Tensor<float> &compute_derivative(Tensor<float> &tensor) const override { return tensor; }

	  protected:
		void computeEncoding(const Tensor<float> &tensor, Tensor<float> &output) {
			switch (this->mode) {
			case OutputMode::OneHot:
				createOneHotEncoding(tensor, output);
				break;
			case OutputMode::Count:
				createOneHotEncoding(tensor, output);
				break;
			default:
				// Invalid state.
				break;
			}
		}

		void createOneHotEncoding(const Tensor<float> &tensor, Tensor<float> &output) {}

		void createCountEncoding(const Tensor<float> &tensor, Tensor<float> &output) {}

	  private:
		Layer<DType> *input;
		std::vector<Layer<DType> *> outputs;

		size_t nrTokens;
		OutputMode mode;
	};
} // namespace Ritsu