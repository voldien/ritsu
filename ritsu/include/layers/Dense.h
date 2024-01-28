#pragma once
#include "../Random.h"
#include "../core/Initializers.h"
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
		Dense(uint32_t units, bool use_bias = true,
			  const Initializer<DType> &weight_init = RandomNormalInitializer<DType>(),
			  const Initializer<DType> &bias_init = RandomNormalInitializer<DType>(), const std::string &name = "dense")
			: Layer(name) {

			/*	*/
			this->units = units;
			/*	*/
			this->shape = {this->units};

			/*	*/
			if (use_bias) {
				this->bias = Tensor<float>(Shape<IndexType>({units}));
			}
		}

		Tensor<float> operator<<(const Tensor<float> &tensor) override {

			Tensor<float> output({this->units}, DTypeSize);

			/*	Verify shape.	*/

			this->compute(tensor, output);

			return output;
		}

		Tensor<float> &operator<<(Tensor<float> &tensor) override {

			/*	Verify shape.	*/

			Tensor<float> inputCopy = tensor;
			this->compute(inputCopy, tensor);

			return tensor;
		}

		Tensor<float> operator>>(Tensor<float> &tensor) override { return tensor; }

		Tensor<float> &operator()(Tensor<float> &tensor) override { return tensor; }

		template <class U> auto &operator()(U &layer) {

			this->setInputs({&layer});
			layer.setOutputs({this});

			this->build(layer.getShape());

			return *this;
		}

		Tensor<float> *getTrainableWeights() noexcept override { return &this->weight; }
		Tensor<float> *getVariables() noexcept override { return &this->bias; }

		void build(const Shape<IndexType> &shape) override {

			/*	Validate */

			/*	*/
			Shape<IndexType> weightShape = Shape<IndexType>({static_cast<IndexType>(this->units), shape[0]});
			this->weight = Tensor<float>(weightShape, this->DTypeSize);

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

		Tensor<float> compute_derivative(const Tensor<float> &tensor) override {
			Tensor<float> output(this->weight.getShape());
			computeDerivative(tensor, output);
			return output;
		}

		Tensor<float> &compute_derivative(Tensor<float> &tensor) const override {
			computeDerivative(tensor, tensor);
			return tensor;
		}

	  private:
		/*	*/
		Layer<DType> *input;
		std::vector<Layer<DType> *> outputs;

	  protected:
		// operator
		void compute(const Tensor<float> &inputTesnor, Tensor<float> &output) {}

		void computeDerivative(const Tensor<float> &error, Tensor<float> &result) const {
			result = computeMatrix(this->weight, error);
			//	result = this->weight * -1.0f;//TODO: validate
		}

		// TODO relocate, and make sure it works.
		// TODO relocate
		Tensor<float> computeMatrix(const Tensor<float> &TensorA, const Tensor<float> &TensorB) const {

			Tensor<float> output(this->getShape());

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
#pragma omp parallel for simd shared(weight)

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
		Tensor<float> bias;
		uint32_t units;
		Tensor<float> weight;
	};

} // namespace Ritsu