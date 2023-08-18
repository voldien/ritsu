#pragma once
#include "../Random.h"
#include "Layer.h"
#include <cassert>
#include <cstddef>
#include <ctime>
#include <random>
#include <vector>

namespace Ritsu {

	class Dense : public Layer<float> {
		//  kernel_initializer='glorot_uniform',
		//    bias_initializer='zeros',

	  public:
		Dense(uint32_t units, bool use_bias = true, const std::string &name = "dense") : Layer(name) {

			/*	*/
			this->units = units;
			/*	*/
			this->shape = {this->units};

			if (use_bias) {
				this->bias = Tensor({units}, DTypeSize);
				this->initbias();
			}
		}

	  public:
		Tensor operator<<(const Tensor &tensor) override {

			Tensor output({this->units}, DTypeSize);

			/*	Verify shape.	*/

			this->compute(tensor, output);

			return output;
		}

		Tensor &operator<<(Tensor &tensor) override {

			/*	Verify shape.	*/

			Tensor inputCopy = tensor;
			this->compute(inputCopy, tensor);

			return tensor;
		}

		Tensor operator>>(Tensor &tensor) override { return tensor; }

		Tensor &operator()(Tensor &tensor) override { return tensor; }

		template <class U> auto &operator()(U &layer) {

			this->setInputs({&layer});
			layer.setOutputs({this});

			return *this;
		}

		Tensor *getTrainableWeights() override { return &this->weight; }
		Tensor *getVariables() override { return &this->bias; }

		void build(const Shape<IndexType> &shape) override {

			/*	Validate */

			/*	*/
			this->weight = Tensor({static_cast<IndexType>(this->units), shape[0]}, this->DTypeSize);

			/*	*/
			this->initweight();
			this->initbias();
		}

		void setOutputs(const std::vector<Layer<DType> *> &layers) override {

			/*	Set input layer */
			this->outputs = layers;

			// TODO verify flatten

			/*	*/
			this->build(this->getInputs()[0]->getShape());
		}

		void setInputs(const std::vector<Layer<DType> *> &layers) override {

			// TODO verify flatten
			Layer<DType> *inputLayer = layers[0];
			if (inputLayer->getShape().getNrDimensions() == 1) {
			}

			this->input = layers[0];

			assert(layers.size() == 1);
		}

		// input

		std::vector<Layer<DType> *> getInputs() const override { return {input}; }
		std::vector<Layer<DType> *> getOutputs() const override { return outputs; }

		Tensor compute_derivative(const Tensor &tensor) override { return tensor; }
		Tensor &compute_derivative(Tensor &tensor) const override { return tensor; }

	  private:
		/*	*/
		Layer<DType> *input;
		std::vector<Layer<DType> *> outputs;

	  protected:
		// operator
		void compute(const Tensor &input, Tensor &output) {
			/*	*/
			// TODO improve
			//#pragma omp parallel
			// Verify the shape.

			const size_t inputDims = input.getShape()[0];

			for (size_t dim = 0; dim < units; dim++) {

				/*	*/
				DType res = DType(0);

#pragma omp parallel for reduction(+ : res) shared(weight, input)
				for (size_t elementIndex = 0; elementIndex < inputDims; elementIndex++) {
					/*	*/
					res +=
						input.getValue<DType>(elementIndex) * this->weight.getValue<DType>(dim * units + elementIndex);
				}
				/*	*/
				if (!this->bias.getNrElements()) {
					res += bias.getValue<DType>(dim);
				}

				/*	*/
				output.getValue<DType>(dim) = res;
			}

			/*	Sum bias.*/
			assert(this->getShape() == output.getShape());
		}

		void initweight() noexcept {

			RandomNormal<DType> random(0.1, 0.2);
#pragma omp parallel shared(weight)
#pragma omp simd

			for (size_t i = 0; i < this->weight.getNrElements(); i++) {
				this->weight.getValue<DType>(i) = random.rand();
			}
		}

		void initbias() noexcept {

			RandomNormal<DType> random(0.1, 0.2);

#pragma omp parallel shared(bias)
#pragma omp simd
			for (size_t i = 0; i < this->bias.getNrElements(); i++) {
				this->bias.getValue<DType>(i) = random.rand();
			}
		}

	  private:
		Tensor bias;
		uint32_t units;
		Tensor weight;
	};

} // namespace Ritsu