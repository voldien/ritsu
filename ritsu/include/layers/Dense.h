#pragma once
#include "Layer.h"
#include <cstddef>
#include <ctime>
#include <random>
#include <vector>

namespace Ritsu {

	class Dense : public Layer<float> {
		//  kernel_initializer='glorot_uniform',
		//    bias_initializer='zeros',

	  public:
		Dense(uint32_t units, bool use_bias = true, const std::string &name = "") : Layer(name) {

			/*	*/
			this->units = units;
			this->shape = {1, this->units};

			if (use_bias) {
				this->bias.resizeBuffer({units}, 4);
				this->initbias();
			}
		}

		Tensor operator<<(const Tensor &tensor) override {

			Tensor output({this->units, 1}, DTypeSize);

			this->compute(tensor, output);

			return output;
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

			/*	*/
			this->weight.resizeBuffer({(unsigned int)this->units * this->getOutputs()[0]->getShape()[1]},
									  this->weight.DTypeSize);

			/*	*/
			this->initweight();
			this->initbias();
		}

		void setInputs(const std::vector<Layer<DType> *> &layers) override { this->input = layers[0]; }

		// input

		std::vector<Layer<DType> *> getInputs() const override { return {input}; }
		std::vector<Layer<DType> *> getOutputs() const override { return outputs; }

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
			for (size_t dim = 0; dim < input.getShape()[0]; dim++) {

				/*	*/
				float res = 0;

#pragma omp parallel for reduction(+ : res) shared(weight, input)
				for (size_t elementIndex = 0; elementIndex < units; elementIndex++) {
					res +=
						input.getValue<DType>(elementIndex) * this->weight.getValue<DType>(dim * units + elementIndex);
				}
				/*	*/
				if (!this->bias.getNrElements()) {
					res += bias.getValue<DType>(dim);
				}

				/*	*/
				output[{(Tensor::IndexType)dim}] = res;
			}
			/*	Sum bias.*/
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