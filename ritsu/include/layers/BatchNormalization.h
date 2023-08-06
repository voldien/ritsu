#pragma once
#include "Layer.h"
#include "Math.h"
#include "Tensor.h"

namespace Ritsu {
	class BatchNormalization : public Layer<float> {
	  public:
		BatchNormalization(const std::string &name = "batchnormalization") : Layer<float>(name) {}

	  private:
		void compute(const Tensor &input, Tensor &output) {

			size_t ndims = 10;
			for (size_t i = 0; i < ndims; i++) {
				Tensor subset = input.getSubset<Tensor>(0, 12);
				DType mean = Math::mean(subset.getRawData<DType>(), subset.getNrElements());
				// TODO add // (subset - mean) /
				(Math::variance<DType>(subset.getRawData<DType>(), subset.getNrElements(), mean) + 0.00001);
			}

			/*	*/
		}
	};
} // namespace Ritsu