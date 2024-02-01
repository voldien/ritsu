#pragma once
#include "Tensor.h"
#include <functional>
#include <iostream>

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	// TODO add template.
	// template <typename T>
	class Loss {
	  public:
		// static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
		//			  "Must be a decimal type(float/double/half) or integer.");
		//
		// using IndexType = unsigned int;
		// static constexpr size_t IndexTypeSize = sizeof(IndexType);
		// using DType = T;
		// const size_t DTypeSize = sizeof(DType);

	  public:
		using LossFunction = void (*)(const Tensor<float> &evoluated, const Tensor<float> &expected,
									  Tensor<float> &output_result);

		Loss() = default;
		//	template <typename T>
		Loss(LossFunction lambda, const std::string &name = "loss") : name(name) { this->loss_function = lambda; }

		virtual Tensor<float> computeLoss(const Tensor<float> &inputX0, const Tensor<float> &inputX1) {

			Tensor<float> batchLossResult(inputX0.getShape(), Tensor<float>::DTypeSize);

			this->loss_function(inputX0, inputX1, batchLossResult);

			/*	Compute mean per each element in batch.	*/

			return batchLossResult;
		}

		virtual Tensor<float> operator()(const Tensor<float> &inputX0, const Tensor<float> &inputX1) {
			return this->computeLoss(inputX0, inputX1);
		}

	  private:
		LossFunction loss_function;
		std::string name;
	};

	static void loss_mse(const Tensor<float> &evoluated, const Tensor<float> &expected, Tensor<float> &output_result) {
		/*	*/
		if (!Tensor<float>::verifyShape(evoluated, expected)) {
			std::cerr << "Bad Shape " << evoluated.getShape() << " not equal " << expected.getShape() << std::endl;
		}

		output_result = std::move(evoluated - expected);
		output_result = std::move(output_result * output_result);
		
		output_result = Tensor<float>::mean<float>(output_result, evoluated.getShape().getAxisDimensions(-1));
	}

	static void loss_msa(const Tensor<float> &evoluated, const Tensor<float> &expected, Tensor<float> &output_result) {
		output_result = evoluated;
		output_result = output_result - expected;
		// TODO add absolute.
		output_result = Tensor<float>::abs(output_result * output_result);
	}

	static void loss_binary_cross_entropy(const Tensor<float> &evoluated, const Tensor<float> &expected,
										  Tensor<float> &output) {
		output = std::move(expected * Tensor<float>::log10(evoluated) * -1.0f);
	}

	static void loss_cross_entropy(const Tensor<float> &evoluated, const Tensor<float> &expected,
								   Tensor<float> &output) {
		Tensor<float> A = Tensor<float>::log10(expected);
		// TODO add support for primitve
		Tensor<float> one(evoluated.getShape());
		output = -expected * A + (one - expected) * A;
	}

	// TODO convert to one shot vector.
	static void loss_cross_catagorial_entropy(const Tensor<float> &evoluated, const Tensor<float> &expected,
											  Tensor<float> &output) {
		output = std::move(expected * Tensor<float>::log10(evoluated) * -1.0f);

		/*Tensor<float> A = inputA * log(inputB);*/
	}

	static void sparse_categorical_crossentropy(const Tensor<float> &evoluated, const Tensor<float> &expected,
												Tensor<float> &output) {

		Tensor<float> expected_one_shot = Tensor<float>::zero(evoluated.getShape());

		// expected_one_shot.getValue<float>((uint32_t)expected.getValue<float>(0)) = 1;

		/*	*/
		if (!Tensor<float>::verifyShape(evoluated, expected_one_shot)) {
			std::cerr << "Sparse Categorical Crossentropy - Bad Shape " << expected_one_shot.getShape() << " not equal "
					  << evoluated.getShape() << std::endl;
		}

		return loss_cross_catagorial_entropy(evoluated, expected_one_shot, output);
	}

	static void loss_ssim(const Tensor<float> &inputA, const Tensor<float> &inputB, Tensor<float> &output) {
		/*Tensor<float> A = inputA * log(inputB);*/
	}

	static void loss_psnr(const Tensor<float> &inputA, const Tensor<float> &inputB, Tensor<float> &output) {

		Tensor<float> &diff = (inputA - inputB).flatten();

		float rmse = std::sqrt(Tensor<float>::mean<float>(diff.pow(2.0f)));

		// TODO:
		// output = 20 * std::log10(255.0 / rmse);
	}
}; // namespace Ritsu
