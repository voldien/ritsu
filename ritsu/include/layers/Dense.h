#pragma once
#include "../Random.h"
#include "Layer.h"
#include "Tensor.h"
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

			/*	*/
			if (use_bias) {
				this->bias = Tensor({units}, DTypeSize);
			}
		}

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


			this->build(layer.getShape());

			return *this;
		}

		Tensor *getTrainableWeights() noexcept override { return &this->weight; }
		Tensor *getVariables() noexcept override { return &this->bias; }

		void build(const Shape<IndexType> &shape) override {

			/*	Validate */

			/*	*/
			Shape<IndexType> weightShape = Shape<IndexType>({static_cast<IndexType>(this->units), shape[0]});
			this->weight = Tensor(weightShape, this->DTypeSize);

			/*	*/
			this->initweight();
			this->initbias();

			/*	*/
			assert(this->weight.getShape().getNrDimensions() == 2);
			assert(this->weight.getShape()[1] == shape[0]);
			assert(this->weight.getShape()[0] == this->units);
		}

		void setOutputs(const std::vector<Layer<DType> *> &layers) override {

			/*	Set input layer */
			this->outputs = layers;

			// TODO verify flatten
			/*	*/
			Layer<DType> *layer = this->getInputs()[0];
		}

		void setInputs(const std::vector<Layer<DType> *> &layers) override {

			// TODO verify flatten
			Layer<DType> *inputLayer = layers[0];
			if (inputLayer->getShape().getNrDimensions() == 1) {
			}

			this->input = inputLayer;

			assert(layers.size() == 1);
		}

		// input

		std::vector<Layer<DType> *> getInputs() const override { return {this->input}; }
		std::vector<Layer<DType> *> getOutputs() const override { return this->outputs; }

		Tensor compute_derivative(const Tensor &tensor) override {
			Tensor output(this->weight.getShape());
			computeDerivative(tensor, output);
			return output;
		}
		Tensor &compute_derivative(Tensor &tensor) const override {
			// computeDerivative(tensor, tensor);
			return tensor;
		}

	  private:
		/*	*/
		Layer<DType> *input;
		std::vector<Layer<DType> *> outputs;

	  protected:
		// operator
		void compute(const Tensor &inputTesnor, Tensor &output) {
			/*	*/
			// assert(inputTesnor.getShape() == this->bias.getShape());
			// TODO improve
			//#pragma omp parallel
			// Verify the shape.

			// TODO matrix multiplication
			output = computeMatrix(this->weight, inputTesnor) + this->bias;
		}

		void computeDerivative(const Tensor &error, Tensor &result) {
			result = this->weight * -1.0f; // computeMatrix(this->weight, error);
		}

		// TODO relocate
		Tensor computeMatrix(const Tensor &TensorA, const Tensor &TensorB) {

			Tensor output(this->getShape());

			for (size_t y = 0; y < TensorA.getShape()[0]; y++) {
				DType sum = 0;
				for (size_t x = 0; x < TensorA.getShape()[1]; x++) {

					size_t index = y * TensorA.getShape()[0] + x;
					sum = TensorA.getValue<DType>(index) * TensorB.getValue<DType>(y);
				}
				output.getValue<DType>(y) = sum;
			}
			return output;
		}

		void initweight() noexcept {
			// TODO improve
			RandomNormal<DType> random(0.1, 1.0);
#pragma omp parallel shared(weight)
#pragma omp simd

			for (size_t i = 0; i < this->weight.getNrElements(); i++) {
				this->weight.getValue<DType>(i) = random.rand();
			}
		}

		void initbias() noexcept {
			// TODO improve
			RandomNormal<DType> random(0.1, 1.0);

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