#pragma once
#include "Loss.h"
#include "Metric.h"
#include "Tensor.h"
#include "core/Shape.h"
#include "layers/Layer.h"
#include "optimizer/Optimizer.h"
#include <cassert>
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
		Model(const std::vector<Layer<T> *> inputs, const std::vector<Layer<T> *> outputs)
			: inputs(inputs), outputs((outputs)) {
			// TODO build up memory.
			/*	TODO other optimization.	*/
			this->build(this->inputs, this->outputs);
		}

		// operator
		// TODO add array.
		void fit(const size_t epochs, const Tensor &inputData, const Tensor &expectedData, const size_t batch_size = 1,
				 const float validation_split = 0.0f, const bool shuffle = false, const bool verbose = true) {

			if (!this->is_build()) {
				/*	Invalid state.	*/
			}

			/*	*/
			const size_t nrTrainBatches = inputData.getShape()[0] / batch_size;
			const size_t nrValidationBatches = 0;
			// TODO verify shape and etc.

			// TODO add array support.
			Tensor batchResult;

			// TODO add support

			const size_t batchXIndex = inputData.getShape()[-1];
			const size_t batchYIndex = expectedData.getShape()[-1];

			Tensor validationData;
			Tensor validationExpected;

			/*	*/
			const Shape<Tensor::IndexType> dataShape =
				inputData.getShape()(batch_size, inputData.getShape().getNrDimensions());
			const Shape<Tensor::IndexType> expectedShape =
				expectedData.getShape()(batch_size, expectedData.getShape().getNrDimensions());
			// Shape::computeNrElements(X.getShape()){

			/*	*/
			const size_t batchDataElementSize = dataShape.getNrElements();
			const size_t batchExpectedElementSize = expectedShape.getNrElements();

			for (size_t nthEpoch = 0; nthEpoch < epochs; nthEpoch++) {
				std::cout << "Epoch: " << nthEpoch << " / " << epochs << std::endl << std::flush;

				// TODO save forward result to the back propegattion.
				for (size_t ibatch = 0; ibatch < nrTrainBatches; ibatch++) {

					/*	Extract subset of the data.	*/
					const Tensor subsetBatchX = std::move(inputData.getSubset<Tensor>(
						ibatch * batch_size * batchDataElementSize, (ibatch + 1) * batch_size * batchDataElementSize));

					const Tensor subsetExpecetedBatch =
						std::move(expectedData.getSubset<Tensor>(ibatch * batch_size * batchExpectedElementSize,
																 (ibatch + 1) * batch_size * batchExpectedElementSize));

					/*	Compute network forward.	*/
					this->forwardPropgation(subsetBatchX, batchResult, batch_size);

					/*	Compute the loss/cost.	*/
					Tensor loss_error = std::move(this->lossFunction.computeLoss(batchResult, subsetExpecetedBatch));

					for (size_t m_index = 0; m_index < this->metrics.size(); m_index++) {
						this->metrics[m_index]->update_state(loss_error);
					}

					/*	*/
					this->backPropagation(loss_error);

					/*	*/
					const DType averageCost = static_cast<float>(Math::mean<DType>(
						loss_error.getRawData<DType>(), static_cast<DType>(loss_error.getNrElements())));

					const DType accuracy = 0;

					this->print_status(std::cout);

					for (size_t m_index = 0; m_index < this->metrics.size(); m_index++) {
						std::cout << Tensor::mean<DType>(this->metrics[m_index]->result());
					}

					std::cout << "\r"
							  << "Batch: " << ibatch << "/" << nrTrainBatches << " - loss: " << averageCost
							  << " -  accuracy: " << accuracy << std::flush;
				}

				/*	Validation Pass.	*/
				for (size_t ibatch = 0; ibatch < nrValidationBatches; ibatch++) {

					/*	Extract subset of the data.	*/
					const Tensor subsetBatchX = std::move(inputData.getSubset<Tensor>(
						ibatch * batch_size * batchDataElementSize, (ibatch + 1) * batch_size * batchDataElementSize));

					const Tensor subsetExpecetedBatch =
						std::move(expectedData.getSubset<Tensor>(ibatch * batch_size * batchExpectedElementSize,
																 (ibatch + 1) * batch_size * batchExpectedElementSize));

					/*	Compute network forward.	*/
					this->forwardPropgation(subsetBatchX, batchResult, batch_size);
					const DType validationCost = 0;
					const DType validationAccuracy = 0;
					/*	*/
				}

				std::cout << std::endl << std::flush;
			}
		}

		Tensor predict(const Tensor &inputTensor, const size_t batch = 1, const bool verbose = false) {

			Tensor result;

			this->forwardPropgation(inputTensor, result, batch);

			return result;
		}

		void compile(Optimizer<T> *optimizer, const Loss loss, const std::vector<Metric *> &compile_metrics = {}) {
			this->optimizer = optimizer;
			this->lossFunction = loss;

			/*	Compile metrics.	*/
			this->metrics = compile_metrics;
		}

		virtual std::string summary() const {
			std::stringstream _summary;

			Layer<T> *current = nullptr;

			for (auto it = this->forwardSequence.begin(); it != this->forwardSequence.end(); it++) {
				Layer<T> *current = (*it);

				_summary << current->getName() << '\t' << " " << current->getShape() << std::endl;
			}
			_summary << "number of weights: " << std::to_string(this->nr_weights) << std::endl;
			_summary << "Trainable in Bytes: " << std::to_string(this->weightSizeInBytes);
			return _summary.str();
		}

		virtual void save(const std::string &path) {
			/*	*/
			/*	*/
		}

		virtual void load(const std::string &path) {
			/*	*/
			/*	*/
		}

		virtual void saveWeight(const std::string &path) {
			/*	*/
			/*	*/
		}

		virtual void loadWeight(const std::string &path) {
			/*	*/
			/*	*/
		}

	  protected:
		// TODO add support for multiple input data and result data..
		void forwardPropgation(const Tensor &inputData, Tensor &result, size_t batchSize) {

			Layer<T> *current = this->inputs[0];
			Tensor res = std::move(inputData); // std::move((*current) << (inputData));

			for (auto it = this->forwardSequence.begin(); it != this->forwardSequence.end(); it++) {
				Layer<T> *current = (*it);

				/*	*/
				// std::cout << current->getName() << " " << current->getShape() << std::endl;

				if (res.getShape() != current->getShape()) {
				}

				res = std::move((*current) << ((const Tensor &)res));
				// Verify shape

				// std::cout << "Result Tensor Shape"
				//		  << " " << res.getShape() << std::endl;
			}
			result = std::move(res);
		}

		void backPropagation(const Tensor &result) {

			// Loss function compute differences.
			Layer<T> *current = nullptr;

			Tensor differental_gradient(result.getShape(), 4);
			differental_gradient = result;

			for (auto it = this->forwardSequence.rbegin(); it != this->forwardSequence.rend(); it++) {

				Layer<T> *current = (*it);

				differental_gradient = current->compute_derivative(static_cast<const Tensor &>(differental_gradient));

				/*	Only apply if */
				Tensor *train_variables = current->getTrainableWeights();
				if (train_variables) {
					this->optimizer->update_step(differental_gradient, *train_variables);
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

		void print_status(std::ostream &stream) {

			//	std::cout << "\r"
			//			  << " Batch: " << ibatch << "/" << nrTrainBatches << " - loss: " << averageCost
			//			  << " -  accuracy: " << accuracy << std::flush;
		}

		bool is_build() const noexcept { return true; }

	  protected:
		std::vector<Layer<T> *> inputs;
		std::vector<Layer<T> *> outputs;
		Optimizer<T> *optimizer;
		Loss lossFunction;
		std::map<std::string, Layer<T> *> layers;

		/*	*/
		std::map<std::string, Tensor> history;
		std::vector<Metric *> metrics;
		// std::map<std::string, Tensor> metrics;

	  private: /*	Internal data.	*/
		std::list<Layer<DType> *> forwardSequence;
		size_t nr_weights;
		size_t weightSizeInBytes;
	};
} // namespace Ritsu