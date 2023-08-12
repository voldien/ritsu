#pragma once
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

		void setOutputs(const std::vector<Layer<DType> *> &layers) override {
			/*	Set input layer */
			this->outputs = layers;

			// TODO verify flatten

			/*	*/
			this->weight = Tensor({(unsigned int)this->units, this->getInputs()[0]->getShape()[0]}, this->DTypeSize);

			/*	*/
			this->initweight();
			this->initbias();
		}

		void setInputs(const std::vector<Layer<DType> *> &layers) override { this->input = layers[0]; }

		// input

		std::vector<Layer<DType> *> getInputs() const override { return {input}; }
		std::vector<Layer<DType> *> getOutputs() const override { return outputs; }

		Tensor compute_deriviate(const Tensor &tensor) override { return tensor; }
		Tensor &compute_deriviate(Tensor &tensor) const override { return tensor; }

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
#pragma omp parallel shared(weight)
#pragma omp simd
			for (size_t i = 0; i < this->weight.getNrElements(); i++) {
				this->weight.getValue<DType>(i) = static_cast<double>(std::rand()) / RAND_MAX * 10.0f;
			}
		}

		void initbias() noexcept {
			std::srand(std::time(nullptr));

#pragma omp parallel shared(bias)
#pragma omp simd
			for (size_t i = 0; i < this->bias.getNrElements(); i++) {
				this->bias.getValue<DType>(i) = static_cast<double>(std::rand()) / RAND_MAX * 10.0f;
			}
		}

	  private:
		Tensor bias;
		uint32_t units;
		Tensor weight;
	};

} // namespace Ritsu