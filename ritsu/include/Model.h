#pragma once
#include "Loss.h"
#include "Tensor.h"
#include "layers/Layer.h"
#include "optimizer/Optimizer.h"
#include <cstddef>
#include <istream>
#include <list>
#include <map>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>

namespace Ritsu {

	template <typename T = float> class Model {
	  public:
		using DType = T;
		const size_t DTypeSize = sizeof(DType);

	  public:
		Model(std::vector<Layer<T> *> inputs, std::vector<Layer<T> *> outputs) : inputs(inputs), outputs((outputs)) {
			// TODO build up memory.
			/*	TODO other optimization.	*/
			this->build(this->inputs, this->outputs);
		}

		// operator
		// TODO add array.
		void fit(size_t epochs, const Tensor &X, const Tensor &Y, size_t batch = 1, bool verbose = true) {

			/*	*/
			const size_t nrBatches = X.getShape()[0] / batch;
			// TODO verify shape and etc.

			// TODO add array support.
			Tensor batchResult;

			// TODO add support
			const size_t batchElementSize = 32 * 32 * 1;
			const size_t batchXIndex = 0;
			const size_t batchYIndex = 0;
			// Shape::computeNrElements(X.getShape()){

			//}

			for (size_t nthEpoch = 0; nthEpoch < epochs; nthEpoch++) {

				for (size_t ibatch = 0; ibatch < nrBatches; ibatch++) {

					/*	Extract subset of the data.	*/
					const Tensor subsetBatchX = std::move(X.getSubset<Tensor>(ibatch * batch * batchElementSize,
																			  (ibatch + 1) * batch * batchElementSize));
					const Tensor subsetBatchY = std::move(Y.getSubset<Tensor>(ibatch * batch * batchElementSize,
																			  (ibatch + 1) * batch * batchElementSize));

					/*	Compute network.	*/
					this->forwardPropgation(subsetBatchX, batchResult, batch);

					/*	Compute the loss/cost.	*/
					// batchResult.Reshape()
					Tensor loss_error = std::move(this->lossFunction.computeLoss(batchResult, subsetBatchY));

					this->backPropagation(loss_error);

					std::cout << '\r' << "Epoch" << nthEpoch << "/" << epochs << " batch: " << ibatch << "/" << ibatch
							  << std::endl;
				}
			}
		}

		Tensor predict(const Tensor &X, size_t batch = 1, bool verbose = false) {

			Tensor result;
			this->forwardPropgation(X, result, batch);
			return result;
		}

		void compile(Optimizer<T> *optimizer, Loss loss) {
			this->optimizer = optimizer;
			this->lossFunction = loss;
		}

		virtual std::string summary() const {
			std::stringstream _summary;

			Layer<T> *current = this->inputs[0];

			for (auto it = this->forwardSequence.begin(); it != this->forwardSequence.end(); it++) {
				Layer<T> *current = (*it);

				_summary << current->getName() << '\t' << " " << current->getShape() << std::endl;
			}
			_summary << "number of weights: " << std::to_string(this->nr_weights) << std::endl;
			_summary << "Trainable in Bytes: " << std::to_string(this->weightSizeInBytes);
			return _summary.str();
		}

	  protected:
		void forwardPropgation(const Tensor &inputData, Tensor &result, size_t batchSize) {

			Layer<T> *current = this->inputs[0];
			Tensor res = std::move(inputData); // std::move((*current) << (inputData));

			for (auto it = this->forwardSequence.begin(); it != this->forwardSequence.end(); it++) {
				Layer<T> *current = (*it);

				/*	*/
				std::cout << current->getName() << " " << current->getShape() << std::endl;

				if (res.getShape() != current->getShape()) {
				}
				res = std::move((*current) << ((const Tensor &)res));

				std::cout << "Result Tensor Shape"
						  << " " << res.getShape() << std::endl;
			}
			result = std::move(res);
		}

		void backPropagation(const Tensor &result) {

			// Loss function compute differences.
			Layer<T> *current = nullptr;

			Tensor differental_gradient(result.getShape(), 4);
			differental_gradient.assignInitValue(1.0f);

			for (auto it = this->forwardSequence.rbegin(); it != this->forwardSequence.rend(); it++) {

				Layer<T> *current = (*it);

				std::cout << current->getName() << std::endl;
				differental_gradient = current->compute_derivative(differental_gradient) *
									   static_cast<DType>(this->optimizer->getLearningRate());

				/*	Only apply if */
				Tensor *train_variables = current->getTrainableWeights();
				if (train_variables != nullptr) {
					*train_variables = differental_gradient;
				}
			}
		}

		virtual void build(std::vector<Layer<T> *> inputs, std::vector<Layer<T> *> outputs) { /*	*/

			// Iterate through each and extract number of trainable variables.
			this->build_sequence(inputs, outputs);

			for (auto it = this->forwardSequence.rbegin(); it != this->forwardSequence.rend(); it++) {
				const std::string &name = (*it)->getName();
				if (this->layers.find(name) != this->layers.end()) {
					(*it)->setName(name + "_1");
				}
				this->layers[(*it)->getName().c_str()] = (*it);
			}
		}

		void build_sequence(const std::vector<Layer<T> *> inputs, const std::vector<Layer<T> *> outputs) {
			// Iterate through each and extract number of trainable variables.
			Layer<T> *current = inputs[0];
			this->nr_weights = 0;
			this->weightSizeInBytes = 0;

			while (current != nullptr) {

				forwardSequence.push_back(current);

				if (current->getTrainableWeights() != nullptr) {
					this->nr_weights += current->getTrainableWeights()->getNrElements();
					this->weightSizeInBytes += current->getTrainableWeights()->getNrElements() * current->DTypeSize;
				}
				// TODO add support

				// TODO verify the shape.

				std::vector<Layer<T> *> layers = current->getOutputs();
				/*	*/
				if (layers.size() > 0) {
					current = layers[0];
				} else {
					current = nullptr;
				}
			}
			// this->forwardSequence.reverse();
		}

	  protected:
		std::vector<Layer<T> *> inputs;
		std::vector<Layer<T> *> outputs;
		Optimizer<T> *optimizer;
		Loss lossFunction;
		std::map<std::string, Layer<T> *> layers;

	  private: /*	Internal data.	*/
		std::list<Layer<DType> *> forwardSequence;
		size_t nr_weights;
		size_t weightSizeInBytes;
	};
} // namespace Ritsu