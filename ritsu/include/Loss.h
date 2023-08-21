#pragma once
#include "Tensor.h"
#include <functional>

namespace Ritsu {

	class Loss {

	  public:
		using LossFunction = void (*)(const Tensor &evoluated, const Tensor &expected, Tensor &output_result);

		Loss() = default;
		//	template <typename T>
		Loss(LossFunction lambda, const std::string &name = "loss") : name(name) { this->loss_function = lambda; }
		virtual Tensor computeLoss(const Tensor &inputX0, const Tensor &inputX1) {
			Tensor out(inputX0.getShape(), inputX0.DTypeSize);

			if (!Tensor::verifyShape(inputX0, inputX1)) {
				// std::cout << inputX0.getShape() << inputX1.getShape() << "Bad Shape" << std::endl;
			}

			this->loss_function(inputX0, inputX1, out);
			return out;
		}

	  private:
		LossFunction loss_function;
		std::string name;
	};

	void loss_mse(const Tensor &inputA, const Tensor &inputB, Tensor &output_result) {
		output_result = inputA;
		output_result = output_result - inputB;
		output_result = output_result * output_result;
	}

	void loss_msa(const Tensor &inputA, const Tensor &inputB, Tensor &output_result) {
		output_result = inputA;
		output_result = output_result - inputB;
		// TODO add absolute.
		output_result = output_result * output_result;
	}

	void loss_cross_entropy(const Tensor &inputA, const Tensor &inputB, Tensor &output) {
		/*Tensor A = inputA * log(inputB);*/
	}

	void loss_cross_catagorial_entropy(const Tensor &inputA, const Tensor &inputB, Tensor &output) {
		/*Tensor A = inputA * log(inputB);*/
	}
	void loss_ssim(const Tensor &inputA, const Tensor &inputB, Tensor &output) { /*Tensor A = inputA * log(inputB);*/ }
}; // namespace Ritsu