
#pragma once
#include "Layer.h"
#include <cstdint>
#include <ctime>
#include <vector>

namespace Ritsu {

	class Conv2D : public Layer<float> {

		//  kernel_initializer='glorot_uniform',
		//    bias_initializer='zeros',

	  public:
		Conv2D(uint32_t filters, const std::vector<uint32_t> &kernel_size, const std::vector<uint32_t> &strides,
			   const std::string &padding, bool useBias = false, const std::string &kernel_init = "",
			   const std::string &bias_init = "", const std::string &name = "Conv2D")
			: Layer<float>(name) {

			this->filters = filters;
			this->stride = strides;
			this->kernel = kernel_size;
		}

		Tensor operator<<(const Tensor &tensor) override {

			// Tensor output({this->units, 1}, DTypeSize);
			//
			// this->compute(tensor, output);

			// return output;
			return tensor;
		}

		Tensor operator>>(Tensor &tensor) override {
			this->compute(tensor, tensor);
			return tensor;
		}

		Tensor &operator()(Tensor &tensor) override {
			this->compute(tensor, tensor);
			return tensor;
		}
		template <class U> auto &operator()(U &layer) {

			this->setInputs({&layer});
			layer.setOutputs({this});

			return *this;
		}

		void build(const Shape<IndexType> &shape) override {
			this->initbias();
			this->initKernels();
		}

		void setInputs(const std::vector<Layer<DType> *> &layers) override {}
		void setOutputs(const std::vector<Layer<DType> *> &layers) override {}

		Tensor compute_derivative(const Tensor &tensor) override { return tensor; }
		Tensor &compute_derivative(Tensor &tensor) const override { return tensor; }

	  protected:
		// operator

		void compute(const Tensor &input, Tensor &output) { /*	*/

			const size_t nrFilters = getNrFilters();
			for (size_t i = 0; i < nrFilters; i++){
				
			}
		}

		void initKernels() noexcept {}

		void initbias() noexcept {
			// std::srand(std::time(NULL));
			// for (int i = 0; i < bias.size(); i++) {
			//	bias[i] = static_cast<double>(std::rand()) / RAND_MAX * 10.0f;
			//}
		}

	  protected:
		size_t getNrFilters() const noexcept { return this->filters; }

	  private:
		size_t filters;
		std::vector<DType> bias;
		Tensor _bias;
		Tensor _kernelWeight;
		std::vector<uint32_t> kernel;
		std::vector<uint32_t> stride;
		std::vector<DType> weight;
	};
} // namespace Ritsu