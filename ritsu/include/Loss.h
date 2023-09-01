#pragma once
#include "Tensor.h"
#include <functional>

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	// TODO add template.
	class Loss {
	  public:
		using LossFunction = void (*)(const Tensor &evoluated, const Tensor &expected, Tensor &output_result);

		Loss() = default;
		//	template <typename T>
		Loss(LossFunction lambda, const std::string &name = "loss") : name(name) { this->loss_function = lambda; }

		virtual Tensor computeLoss(const Tensor &inputX0, const Tensor &inputX1) {
			Tensor out(inputX0.getShape(), Tensor::DTypeSize);

			if (!Tensor::verifyShape(inputX0, inputX1)) {
				std::cout << inputX0.getShape() << inputX1.getShape() << "Bad Shape" << std::endl;
			}

			this->loss_function(inputX0, inputX1, out);

			/*	Compute mean per each element in batch.	*/

			return out;
		}

	  private:
		LossFunction loss_function;
		std::string name;
	};

	static void loss_mse(const Tensor &evoluated, const Tensor &expected, Tensor &output_result) {
		output_result = evoluated;
		output_result = output_result - expected;
		output_result = output_result * output_result;
	}

	static void loss_msa(const Tensor &evoluated, const Tensor &expected, Tensor &output_result) {
		output_result = evoluated;
		output_result = output_result - expected;
		// TODO add absolute.
		output_result = Tensor::abs(output_result * output_result);
	}

	static void loss_binary_cross_entropy(const Tensor &evoluated, const Tensor &expected, Tensor &output) {
		output = std::move(expected * Tensor::log10(evoluated) * -1.0f);
	}

	static void loss_cross_entropy(const Tensor &evoluated, const Tensor &expected, Tensor &output) {
		Tensor A = Tensor::log10(expected);
		// TODO add support for primitve
		Tensor one(evoluated.getShape());
		output = -expected * A + (one - expected) * A;
	}

	static void loss_cross_catagorial_entropy(const Tensor &evoluated, const Tensor &expected, Tensor &output) {
		output = std::move(expected * Tensor::log10(evoluated) * -1.0f);

		/*Tensor A = inputA * log(inputB);*/
	}

	void loss_ssim(const Tensor &inputA, const Tensor &inputB, Tensor &output) { /*Tensor A = inputA * log(inputB);*/
	}
}; // namespace Ritsu