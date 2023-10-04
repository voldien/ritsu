#pragma once
#include "../Math.h"
#include "Layer.h"

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class BatchNormalization : public Layer<float> {
	  public:
		BatchNormalization(const std::string &name = "batch normalization") : Layer<float>(name) {}

		Tensor operator<<(const Tensor &tensor) override { return tensor; }

		Tensor &operator<<(Tensor &tensor) override { return tensor; }

		template <class U> auto &operator()(U &layer) {

			this->setInputs({&layer});
			layer.setOutputs({this});

			return *this;
		}

		void setInputs(const std::vector<Layer<DType> *> &layers) override {
			this->shape = layers[0]->getShape();
			/*	Set input layer */
			this->input = layers[0];
		}

		void setOutputs(const std::vector<Layer<DType> *> &layers) override {
			/*	Set input layer */
			this->outputs = layers;

			// TODO verify flatten

			/*	*/
			this->build(this->getInputs()[0]->getShape());
		}
		void build(const Shape<IndexType> &shape) override { /*	Validate */
		}

		std::vector<Layer<DType> *> getInputs() const override { return {input}; }
		std::vector<Layer<DType> *> getOutputs() const override { return outputs; }

		Tensor compute_derivative(const Tensor &tensor) override { return tensor; }
		Tensor &compute_derivative(Tensor &tensor) const override { return tensor; }

		Tensor *getTrainableWeights() override { return nullptr; }
		Tensor *getVariables() override { return nullptr; }

	  private:
		void compute(const Tensor &input, Tensor &output) {

			const size_t ndims = 10;

			for (size_t i = 0; i < ndims; i++) {

				Tensor subset = input.getSubset<Tensor>(0, 12);
				DType mean = Math::mean(subset.getRawData<DType>(), subset.getNrElements());
				// TODO add // (subset - mean) /
				(Math::variance<DType>(subset.getRawData<DType>(), subset.getNrElements(), mean) + 0.00001);
			}

			/*	*/
		}

	  private:
		Tensor beta;
		Tensor alpha;

		/*	*/
		/*	*/
		Layer<DType> *input;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu