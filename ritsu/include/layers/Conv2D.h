
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
			   const std::string &bias_init = "", const std::string &name = "")
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

		Tensor operator>>(Tensor &tensor) override { return tensor; }

		Tensor &operator()(Tensor &tensor) override {
			this->compute(tensor, tensor);
			return tensor;
		}

		void setInputs(const std::vector<Layer<DType> *> &layers) override {}
		void setOutputs(const std::vector<Layer<DType> *> &layers) override {}

		Tensor compute_deriviate(const Tensor &tensor) override { return tensor; }
		Tensor &compute_deriviate(Tensor &tensor) const override { return tensor; }

	  protected:
		// operator

		void compute(const Tensor &input, Tensor &output) { /*	*/
		}

		void initbias() {
			// std::srand(std::time(NULL));
			// for (int i = 0; i < bias.size(); i++) {
			//	bias[i] = static_cast<double>(std::rand()) / RAND_MAX * 10.0f;
			//}
		}

	  private:
		std::vector<DType> bias;
		Tensor _bias;
		Tensor _weight;
		std::vector<uint32_t> kernel;
		std::vector<uint32_t> stride;
		uint32_t filters;
		std::vector<DType> weight;
	};
} // namespace Ritsu