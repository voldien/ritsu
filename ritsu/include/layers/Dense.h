#pragma once
#include "Layer.h"
#include "Tensor.h"
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

			this->units = units;
			this->shape = {1, this->units};

			if (use_bias) {
				this->bias.resize(units);
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

		void setOutputs(const std::vector<Layer<DType> *> &layers) override {
			/*	Set input layer */
			this->outputs = layers;

			this->weight.resize((size_t)this->units * this->getOutputs()[0]->getNrDimension()[1]);
			this->initweight();
		}

		void setInputs(const std::vector<Layer<DType> *> &layers) override { this->input = layers[0]; }

		// input

		std::vector<Layer<DType> *> getInputs() const override { return {input}; }
		std::vector<Layer<DType> *> getOutputs() const override { return outputs; }

		Layer<DType> *input;
		std::vector<Layer<DType> *> outputs;

	  protected:
		// operator

		void compute(const Tensor &input, Tensor &output) {
			/*	*/
			// TODO improve
			//#pragma omp parallel
			for (size_t dim = 0; dim < input.getNrDimension()[0]; dim++) {

				/*	*/
				float res = 0;

#pragma omp parallel for reduction(+ : res) shared(weight, input)
				for (size_t elementIndex = 0; elementIndex < units; elementIndex++) {
					res += input[{(Tensor::IndexType)dim}] * this->weight[dim * units + elementIndex];
				}
				/*	*/
				if (!this->bias.empty()) {
					res += bias[dim];
				}

				/*	*/
				output[{(Tensor::IndexType)dim}] = res;
			}
			/*	Sum bias.*/
		}

		void initweight() noexcept {
#pragma omp parallel shared(weight)
#pragma omp simd
			for (size_t i = 0; i < this->weight.size(); i++) {
				this->weight[i] = static_cast<double>(std::rand()) / RAND_MAX * 10.0f;
			}
		}

		void initbias() noexcept {
			std::srand(std::time(nullptr));

#pragma omp parallel shared(bias)
#pragma omp simd
			for (size_t i = 0; i < this->bias.size(); i++) {
				this->bias[i] = static_cast<double>(std::rand()) / RAND_MAX * 10.0f;
			}
		}

	  private:
		std::vector<DType> bias;
		uint32_t units;
		std::vector<DType> weight;
	};

} // namespace Ritsu