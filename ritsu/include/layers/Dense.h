#pragma once
#include "Layer.h"

namespace Ritsu {

	class Dense : public Layer<float> {

		//  kernel_initializer='glorot_uniform',
		//    bias_initializer='zeros',

	  public:
		Dense(uint32_t units, bool use_bias = true, const std::string &name = "") : Layer(name) {
			if (use_bias) {
				initbias();
			}

			/*	Assign random values.	*/
			weight.resize(units * units);
			bias.resize(units);
		}

		virtual Tensor operator<<(Tensor &tensor) override {
			Tensor output({units, 1},4);
			compute(tensor, output);

			return output;
		}

		virtual Tensor operator>>(Tensor &tensor) override { return tensor; }

		virtual Tensor &operator()(Tensor &tensor) override { return tensor; }

	  protected:
		// operator

		void compute(const Tensor &input, Tensor &output) {
			/*	*/
			for (size_t x = 0; x < input.getNrDimension()[1]; x++) {
				float res = 0;
				for (size_t y = 0; y < units; y++) {
					res += input[{x}] * weight[x * units + y];
				}
				 res +=  bias[x];
				 output[{x,1}] = res;
			}
			/*	Sum bias.*/
		}

		void initbias() {}

	  private:
		std::vector<DType> bias;
		unsigned int units;
		std::vector<DType> weight;
	};
} // namespace Ritsu