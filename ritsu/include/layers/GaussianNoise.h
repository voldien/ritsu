#pragma once
#include "Layer.h"
#include <ctime>
#include <random>

namespace Ritsu {

	class GuassianNoise : public Layer<float> {

	  public:
		GuassianNoise(float stddev, const std::string &name = "") : Layer(name), stddev(stddev) {}

		Tensor operator<<(const Tensor &tensor) override {
			// compute(tensor);
			return tensor;
		}

		Tensor operator>>(Tensor &tensor) override {
			compute(tensor);
			return tensor;
		}

		// virtual Tensor operator()(Tensor &tensor) {
		//	compute(tensor);
		//	return tensor;
		//}

		Tensor &operator()(Tensor &tensor) override {
			compute(tensor);
			return tensor;
		}

	  protected:
		void compute(Tensor &tensor) {
			std::random_device rd;	// Will be used to obtain a seed for the random number engine
			std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
			std::uniform_real_distribution<> dis(1.0, 2.0);
			/*Iterate through each all elements.    */
			const size_t nrElements = tensor.getNrElements();
			for (size_t i = 0; i < nrElements; i++) {
				tensor.getValue<float>(i) = dis(gen);
			}
		}

	  private:
		float stddev;
	};
} // namespace Ritsu