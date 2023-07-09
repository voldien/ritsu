#pragma once
#include "Loss.h"
#include "Tensor.h"
#include "layers/Layer.h"
#include "optimizer/Optimizer.h"
#include <cstddef>

namespace Ritsu {

	template <typename T = float> class Model {

	  public:
		Model(std::vector<Layer<T> *> input, std::vector<Layer<T> *> output) : inputs(input), outputs((output)) {
			// TODO build up memory.
		}

		// operator
		// TODO add array.
		void fit(size_t epochs, const Tensor &X, const Tensor &Y, size_t batch = 1) {
			/*	*/
			size_t nrBatches = X.getNrDimension()[0] / batch;
			// TODO verify shape and etc.

			for (size_t nthEpoch = 0; nthEpoch < epochs; nthEpoch++) {

				for (size_t ibatch = 0; ibatch < nrBatches; ibatch++) {
					const Tensor subsetBatchX = X.getSubset<Tensor>(ibatch * batch, (ibatch + 1) * batch);
					const Tensor subsetBatchY = Y.getSubset<Tensor>(ibatch * batch, (ibatch + 1) * batch);

					Tensor batchResult;

					this->forwardPropgation(subsetBatchX, batchResult);

					Tensor diff = this->lossFunction.computeLoss(subsetBatchX, Y);

					this->backPropagation(diff);
				}
				// Layer<T> *current = this->inputs[0];
				//
				// Tensor res = (*current) << (X);
				//
				// while (current != nullptr) {
				//	// TODO add support
				//	if (current->getOutputs().size() == 0) {
				//		break;
				//	}
				//	std::vector<Layer<T> *> layers = current->getOutputs();
				//	for (size_t i = 0; i < layers.size(); i++) {
				//	}
				//
				//	std::cout << current->getName() << std::endl;
				//	current = layers[0];
				//
				//	res = (*current) << (res);
				//}
				// std::cout << current->getName() << std::endl;
				//
				// Tensor diff = this->lossFunction.computeLoss(res, Y);
				//
				// this->backPropagation(diff);
				// std::cout << std::endl << "Fit complete " << res << std::endl;
			}
		}

		Tensor predict(const Tensor &X) {

			Tensor result;
			this->forwardPropgation(X, result);
			return result;
		}

		// Tensor &operator<<(Tensor &tensor) override {
		//	this->computeActivation(tensor);
		//	return tensor;
		//}
		//
		// Tensor operator>>(Tensor &tensor) override {
		//	this->computeActivation(tensor);
		//	return tensor;
		//}

		void compile(Optimizer<T> *optimizer, Loss loss) {
			this->optimizer = optimizer;
			this->lossFunction = loss;
		}

	  protected:
		void forwardPropgation(const Tensor &inputData, Tensor &result) {

			Layer<T> *current = this->inputs[0];
			Tensor res = (*current) << (inputData);

			while (current != nullptr) {
				// TODO add support
				if (current->getOutputs().size() == 0) {
					break;
				}

				std::vector<Layer<T> *> layers = current->getOutputs();
				for (size_t i = 0; i < layers.size(); i++) {
				}

				std::cout << current->getName() << std::endl;
				current = layers[0];

				res = (*current) << (res);
			}
			result = res;
		}

		void backPropagation(const Tensor &result) {
			// Loss function compute differences.
			Layer<T> *current = this->outputs[0];
			// for(current != nullptr){
			//	std::vector<Layer<T> *> inputs = current->getInputs();
			//	current = inputs[0];
			//
			//}

			// Backprogation.
		}

		virtual void build(std::vector<Layer<T> *> input, std::vector<Layer<T> *> output) {}

	  private:
		std::vector<Layer<T> *> inputs;
		std::vector<Layer<T> *> outputs;
		Optimizer<T> *optimizer;
		Loss lossFunction;

	  private:
	};
} // namespace Ritsu