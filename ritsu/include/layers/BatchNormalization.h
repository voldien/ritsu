#pragma once
#include "Layer.h"
#include "Math.h"
#include "Tensor.h"

namespace Ritsu {
	class BatchNormalization : public Layer<float> {
	  public:
		BatchNormalization(const std::string &name = "batch normalization") : Layer<float>(name) {}

		Tensor operator<<(const Tensor &tensor) override { return tensor; }

		Tensor &operator<<(Tensor &tensor) override { return tensor; }

		void setInputs(const std::vector<Layer<DType> *> &layers) override {}
		void setOutputs(const std::vector<Layer<DType> *> &layers) override {}

		Tensor compute_deriviate(const Tensor &tensor) override { return tensor; }
		Tensor &compute_deriviate(Tensor &tensor) const override { return tensor; }

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

	  private:
		Layer<DType> *input;
		std::vector<Layer<DType> *> outputs;
	};
} // namespace Ritsu