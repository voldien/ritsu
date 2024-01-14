#pragma once
#include "Loss.h"
#include "Metric.h"
#include "Tensor.h"
#include "core/Shape.h"
#include "layers/Layer.h"
#include "optimizer/Optimizer.h"
#include <cassert>
#include <cstddef>
#include <exception>
#include <istream>
#include <list>
#include <map>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace Ritsu {

	/**
	 * @brief
	 *
	 * @tparam T
	 */
	template <typename T = float> class Model {
	  public:
		using DType = T;
		const size_t DTypeSize = sizeof(DType);

	  public:
		Model(const std::vector<Layer<T> *> inputs, const std::vector<Layer<T> *> outputs,
			  const std::string &name = "model")
			: inputs(inputs), outputs((outputs)) {
			// TODO build up memory.
			/*	TODO other optimization.	*/
			this->build(this->inputs, this->outputs);
		}

		// operator
		// TODO add array.
		// TODO add callback.
		void fit(const size_t epochs, const Tensor &inputData, const Tensor &expectedData, const size_t batch_size = 1,
				 const float validation_split = 0.0f, const bool shuffle = false, const bool verbose = true) {

			const size_t batch_shape_index = 0; // TODO fix
			/*	*/
			if (!this->is_build()) {
				/*	Invalid state.	*/
				throw std::bad_exception();
			}

			/*	*/
			const size_t nrTrainBatches = inputData.getShape()[0] / batch_size;
			const size_t nrValidationBatches = 0;
			// TODO verify shape and etc.

			// TODO add array support.
			Tensor batchResult;

			// TODO add support

			/*	Number of batches in dataset.	*/
			const size_t batchXIndex = inputData.getShape()[0];
			const size_t batchYIndex = expectedData.getShape()[0];

			/*	*/
			Tensor validationData;
			Tensor validationExpected;

			/*	Batch Shape Size.	*/
			Shape<Tensor::IndexType> batchDataShape = inputData.getShape();
			batchDataShape[batch_shape_index] = batch_size;
			Shape<Tensor::IndexType> batchExpectedShape = expectedData.getShape();
			batchExpectedShape[batch_shape_index] = batch_size;

			std::map<std::string, Tensor> cachedResult;

			/*	*/
			const size_t batchDataElementSize = batchDataShape.getNrElements();
			const size_t batchExpectedElementSize = batchExpectedShape.getNrElements();

			for (size_t nthEpoch = 0; nthEpoch < epochs; nthEpoch++) {
				std::cout << "Epoch: " << nthEpoch << " / " << epochs << std::endl << std::flush;

				// TODO save forward result to the back propegattion.

				/*	Train pass.	*/
				for (size_t ibatch = 0; ibatch < nrTrainBatches; ibatch++) {

					/*	Extract subset of the data.	*/
					const Tensor subsetBatchX = std::move(
						inputData.getSubset(ibatch * batch_size * batchDataElementSize,
													(ibatch + 1) * batch_size * batchDataElementSize, batchDataShape));

					const Tensor subsetExpecetedBatch = std::move(expectedData.getSubset(
						ibatch * batch_size * batchExpectedElementSize,
						(ibatch + 1) * batch_size * batchExpectedElementSize, batchExpectedShape));

					/*	Compute network forward.	*/
					this->forwardPropgation(subsetBatchX, batchResult, batch_size, &cachedResult);

					/*	Compute the loss/cost.	*/
					Tensor loss_error = std::move(this->lossFunction.computeLoss(batchResult, subsetExpecetedBatch));

					/*	Apply metric update.	*/
					for (size_t m_index = 0; m_index < this->metrics.size(); m_index++) {
						this->metrics[m_index]->update_state(loss_error, batchResult);
					}

					/*	*/
					this->backPropagation(loss_error, cachedResult);

					this->print_status(std::cout);

					std::cout << "\r"
							  << "Batch: " << ibatch << "/" << nrTrainBatches; //<< " - loss: " << averageCost
					//						  << " -  accuracy: " << accuracy;

					for (size_t m_index = 0; m_index < this->metrics.size(); m_index++) {
						std::cout << " - " << this->metrics[m_index]->getName() << ": "
								  << Tensor::mean<DType>(this->metrics[m_index]->result());
					}
					std::cout << std::flush;
				}

				/*	Validation Pass. Only compute, no backpropgation.	*/
				for (size_t ibatch = 0; ibatch < nrValidationBatches; ibatch++) {

					/*	Extract subset of the data.	*/
					const Tensor subsetBatchX = std::move(inputData.getSubset (
						ibatch * batch_size * batchDataElementSize, (ibatch + 1) * batch_size * batchDataElementSize));

					const Tensor subsetExpecetedBatch =
						std::move(expectedData.getSubset (ibatch * batch_size * batchExpectedElementSize,
																 (ibatch + 1) * batch_size * batchExpectedElementSize));

					/*	Compute network forward.	*/
					this->forwardPropgation(subsetBatchX, batchResult, batch_size);
					const DType validationCost = 0;
					const DType validationAccuracy = 0;
					/*	*/
				}

				/*	Update history, using all metrics.	*/
				for (size_t m_index = 0; m_index < this->metrics.size(); m_index++) {
					/*	*/
					this->history[this->metrics[m_index]->getName()].append(
						this->metrics[m_index]->result().getValue<float>(0));
				}

				std::cout << std::endl << std::flush;
			}
		}

		Tensor evaluate(const Tensor &XData, const Tensor &YData, const size_t batch = 1, const bool verbose = false) {

			Tensor result;

			this->forwardPropgation(XData, result, batch);

			return result;
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
			// TODO verify the objects.

			/*	*/
			for (size_t m_index = 0; m_index < this->metrics.size(); m_index++) {
				this->history[this->metrics[m_index]->getName()] = Tensor({1});
			}
		}

		Layer<DType> *getLayer(const std::string &name) { return this->layers[name]; }

		virtual std::string summary() const {
			std::stringstream _summary;

			Layer<T> *current = nullptr;

			/*	*/

			/*	List all layers.	*/
			for (auto it = this->forwardSequence.begin(); it != this->forwardSequence.end(); it++) {
				Layer<T> *current = (*it);

				// TODO get type
				_summary << current->getName() << '\t' << " " << current->getShape() << '\t';

				for (size_t i = 0; i < current->getInputs().size(); i++) {
					if (i == 0) {
						_summary << "<-- [ ";
					}
					_summary << current->getInputs()[i]->getName() << " ";
					if (i == current->getInputs().size() - 1) {
						_summary << "]";
					}
				}

				_summary << std::endl;
			}

			/*	Summary of number of parameters and size.	*/
			_summary << "number of weights: " << std::to_string(this->nr_weights) << std::endl;
			_summary << "Trainable in Bytes: "
					 << std::to_string(static_cast<int>(this->trainableWeightSizeInBytes / 1024.0f)) << " KB"
					 << std::endl;
			_summary << "None-Trainable in Bytes: "
					 << std::to_string(static_cast<int>(this->noneTrainableWeightSizeInBytes / 1024.0f)) << " KB";
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

		virtual Tensor computeLoss(const Tensor &inputData, Tensor &result) { return {}; }

	  protected:
		// TODO add support for multiple input data and result data..
		void forwardPropgation(const Tensor &inputData, Tensor &result, size_t batchSize,
							   std::map<std::string, Tensor> *cacheResult = nullptr) {

			Layer<T> *current = this->inputs[0];
			Tensor layerResult = std::move(inputData); // std::move((*current) << (inputData));

			for (auto it = this->forwardSequence.begin(); it != this->forwardSequence.end(); it++) {
				Layer<T> *current = (*it);

				const bool junctionLayer = is_junction_layer(current);

				for (size_t i = 0; i < batchSize; i++) {
					// TODO fix.

					/*	Perform layer on data.	*/
					layerResult = std::move((*current) << ((const Tensor &)layerResult));

					/*	*/
					auto shape = layerResult.getShape().getSubShape(1);
					if (shape != current->getShape()) {
						/*	*/
						std::cerr << "Invalid Shape: " << shape << " " << current->getShape() << std::endl;
					}

					// std::cout << "Result Tensor Shape"
					//		  << " " << res.getShape() << std::endl;
				}

				/*	Assign result if memory provided.	*/
				if (cacheResult != nullptr) {
					(*cacheResult)[(*current).getName()] = layerResult;
				}
			}
			result = std::move(layerResult);
		}

		void backPropagation(const Tensor &result, std::map<std::string, Tensor> &cacheResult) {

			// Loss function compute differences.
			Layer<T> *current = nullptr;

			Tensor differental_gradient(result.getShape());
			differental_gradient = result;

			/*	*/
			for (auto it = this->forwardSequence.rbegin(); it != this->forwardSequence.rend(); it++) {

				Layer<T> *current = (*it);

				differental_gradient = current->compute_derivative(static_cast<const Tensor &>(differental_gradient));

				/*	Only apply if */
				Tensor *train_variables = current->getTrainableWeights();
				if (train_variables) {
					this->optimizer->update_step(differental_gradient, *train_variables);
				}

				/*	*/
				Tensor &layerResult = cacheResult[current->getName()];

				differental_gradient = differental_gradient; // - (layerResult - differental_gradient);
			}
		}

		virtual void build(std::vector<Layer<T> *> inputs, std::vector<Layer<T> *> outputs) { /*	*/

			// Iterate through each and extract number of trainable variables.
			this->build_sequence(inputs, outputs);

			/*	Extract all layer and rename to make them all unique.	*/
			for (auto it = this->forwardSequence.rbegin(); it != this->forwardSequence.rend(); it++) {
				const std::string &name = (*it)->getName();
				std::string newName = (*it)->getName();

				int index = 0;
				while (this->layers.find(newName) != this->layers.end()) {

					newName = name + "_" + (*it)->getDType().name();
					newName += (char)('0' + (char)index);
				}

				(*it)->setName(newName);
				this->layers[newName] = (*it);
			}
		}

		void build_sequence(const std::vector<Layer<T> *> inputs, const std::vector<Layer<T> *> outputs) {
			// Iterate through each and extract number of trainable variables.
			Layer<T> *current = inputs[0];
			this->nr_weights = 0;
			this->trainableWeightSizeInBytes = 0;
			this->noneTrainableWeightSizeInBytes = 0;

			while (current != nullptr) {

				forwardSequence.push_back(current);

				// TODO add support.
				if (is_junction_layer(current)) {
				}

				if (current->getTrainableWeights() != nullptr) {
					this->nr_weights += current->getTrainableWeights()->getNrElements();
					this->trainableWeightSizeInBytes +=
						current->getTrainableWeights()->getNrElements() * current->DTypeSize;
				}
				if (current->getVariables() != nullptr) {
					// this->nr_weights += current->getVariables()->getNrElements();
					this->noneTrainableWeightSizeInBytes +=
						current->getVariables()->getNrElements() * current->DTypeSize;
				}
				// TODO add support

				// TODO verify the shape.

				std::vector<Layer<T> *> layers = current->getOutputs();
				/*	*/
				if (!layers.empty()) {
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

		bool is_build() const noexcept { return layers.size() > 0; }

		bool is_junction_layer(const Layer<DType> *layer) const noexcept { return layer->getInputs().size() > 1; }

	  protected:
		/*	*/
		std::vector<Layer<T> *> inputs;
		std::vector<Layer<T> *> outputs;

		/*	*/
		Optimizer<T> *optimizer;
		Loss lossFunction;

		/*	*/
		std::map<std::string, Layer<T> *> layers;

		/*	*/
		std::map<std::string, Tensor> history;

		/*	*/
		std::vector<Metric *> metrics;

	  private: /*	Internal data.	*/
		std::list<Layer<DType> *> forwardSequence;
		size_t nr_weights;
		size_t trainableWeightSizeInBytes;
		size_t noneTrainableWeightSizeInBytes;
		std::string name;
	}; // namespace Ritsu
} // namespace Ritsu