/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2023 Valdemar Lindberg
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 */
#pragma once
#include "Loss.h"
#include "Metric.h"
#include "Object.h"
#include "Tensor.h"
#include "core/Shape.h"
#include "core/Time.h"
#include "layers/Layer.h"
#include "optimizer/Optimizer.h"
#include <cassert>
#include <chrono>
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
	template <typename T = float> class Model : public Object {
	  public:
		using DType = T;
		const size_t DTypeSize = sizeof(DType);
		using IndexType = unsigned int;

	  public:
		Model(const std::vector<Layer<T> *> inputs, const std::vector<Layer<T> *> outputs,
			  const std::string &name = "model")
			: Object(name), inputs(inputs), outputs((outputs)) {
			// TODO build up memory.
			/*	TODO other optimization.	*/
			this->build(this->inputs, this->outputs);
		}

		// operator
		// TODO add array.
		// TODO add callback.
		void fit(const size_t epochs, const Tensor<float> &inputData, const Tensor<float> &expectedData,
				 const size_t batch_size = 1, const float validation_split = 0.0f, const bool shuffle = false,
				 const bool verbose = true) {

			const size_t batch_shape_index = 0;

			/*	*/
			if (!this->is_build()) {
				/*	Invalid state.	*/
				throw std::bad_exception();
			}

			/*	Number of batches in dataset.	*/
			const size_t batchXIndex = inputData.getShape()[batch_shape_index];
			const size_t batchYIndex = expectedData.getShape()[batch_shape_index];

			/*	*/
			const size_t nrTrainBatches = std::floor(static_cast<float>(batchXIndex) / static_cast<float>(batch_size));
			const size_t nrValidationBatches = std::floor(nrTrainBatches * validation_split);
			// TODO verify shape and etc.

			// TODO add array support.
			Tensor<float> batchResult;

			// TODO add support

			/*	*/
			Tensor<float> validationData;
			Tensor<float> validationExpected;

			/*	Batch Shape Size.	*/
			Shape<Tensor<float>::IndexType> batchDataShape = inputData.getShape();
			batchDataShape[batch_shape_index] = batch_size;
			Shape<Tensor<float>::IndexType> batchExpectedShape = expectedData.getShape();
			batchExpectedShape[batch_shape_index] = batch_size;

			/*	TODO: setup cache.	*/
			std::map<std::string, Tensor<float>> cachedResult;

			/*	*/
			const size_t batchDataElementSize = batchDataShape.getNrElements();
			const size_t batchExpectedElementSize = batchExpectedShape.getNrElements();

			Time time;
			time.start();

			Tensor<float> timeSample({8});
			uint32_t timeIndex = 0;

			for (size_t nthEpoch = 0; nthEpoch < epochs; nthEpoch++) {
				std::cout << "Epoch: " << nthEpoch << " / " << epochs << std::endl << std::flush;

				/*	Train pass.	*/
				for (size_t ibatch = 0; ibatch < nrTrainBatches; ibatch++) {

					/*	Extract subset of the data.	*/
					const Tensor<float> subsetBatchX =
						inputData.getSubset({{static_cast<unsigned int>(ibatch * batch_size),
											  static_cast<unsigned int>((ibatch + 1) * batch_size) - 1}});

					/*	*/
					const Tensor<float> subsetExpectedBatch =
						expectedData.getSubset({{static_cast<unsigned int>(ibatch * batch_size),
												 static_cast<unsigned int>((ibatch + 1) * batch_size) - 1}});

					/*	Compute network forward.	*/
					this->forwardPropgation(subsetBatchX, batchResult, batch_size, &cachedResult);

					/*	Compute the loss/cost.	*/
					Tensor<float> loss_error =
						std::move(this->lossFunction.computeLoss(batchResult, subsetExpectedBatch));

					/*	Apply metric update.	*/
					for (size_t m_index = 0; m_index < this->metrics.size(); m_index++) {
						this->metrics[m_index]->update_state(loss_error, batchResult);
					}

					/*	*/
					this->backPropagation(loss_error, cachedResult);

					/*	*/
					this->print_status(std::cout);

					{
						/*	*/
						timeSample.getValue<float>(timeIndex) = time.deltaTime<float>();
						timeIndex = (timeIndex + 1) % timeSample.getNrElements();
						const float averageTime = Tensor<float>::mean<float>(timeSample);

						const size_t nrBatchPerSecond = 1.0f / averageTime;
						const float expectedEpochTime = (nrTrainBatches - ibatch) * averageTime;

						using fsec = duration<float>;
						auto p = round<nanoseconds>(fsec{expectedEpochTime});

						std::chrono::duration_cast<std::chrono::minutes>(p);

						std::cout << "\r"
								  << "Batch: " << ibatch << "/"
								  << nrTrainBatches // << " " << nrBatchPerSecond << "batch/Sec"
								  << " ETA: " << std::chrono::duration_cast<std::chrono::minutes>(p).count()
								  << " - lr: " << this->optimizer->getLearningRate();

						for (size_t m_index = 0; m_index < this->metrics.size(); m_index++) {
							std::cout << " - " << this->metrics[m_index]->getName() << ": "
									  << Tensor<float>::mean<DType>(this->metrics[m_index]->result());
						}
					}
					std::cout << std::flush;

					time.update();
				}

				/*	Validation Pass. Only compute, no backpropgation.	*/
				for (size_t ibatch = 0; ibatch < nrValidationBatches; ibatch++) {

					/*	Extract subset of the data.	*/
					const Tensor<float> subsetBatchX =
						inputData.getSubset(ibatch * batch_size * batchDataElementSize,
											(ibatch + 1) * batch_size * batchDataElementSize, batchDataShape);

					const Tensor<float> subsetExpecetedBatch = expectedData.getSubset(
						ibatch * batch_size * batchExpectedElementSize,
						(ibatch + 1) * batch_size * batchExpectedElementSize, batchExpectedShape);

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
						this->metrics[m_index]->result().template getValue<float>(0));
				}

				std::cout << std::endl << std::flush;
			}
		}

		Tensor<float> evaluate(const Tensor<float> &XData, const Tensor<float> &YData, const size_t batch = 1,
							   const bool verbose = false) {

			Tensor<float> result;

			Time time;

			this->forwardPropgation(XData, result, batch);

			return result;
		}

		Tensor<float> predict(const Tensor<float> &inputTensor, const size_t batch = 1, const bool verbose = false) {

			Tensor<float> result;

			Time time;

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
				this->history[this->metrics[m_index]->getName()] = Tensor<float>({1});
			}
		}

		Layer<DType> *getLayer(const std::string &name) { return this->layers[name]; }

		virtual std::string summary() const {
			std::stringstream _summary;

			Layer<T> *current = nullptr;

			/*	*/

			/*	List all layers.	*/
			for (auto it = this->forwardSequence.begin(); it != this->forwardSequence.end(); it++) {

				const Layer<T> *current = (*it);

				// TODO get type
				_summary << current->getName() << '\t' << " " << current->getShape() << '\t';

				/*	*/
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

	  protected:
		// TODO add support for multiple input data and result data..
		void forwardPropgation(const Tensor<float> &inputData, Tensor<float> &result, size_t batchSize,
							   std::map<std::string, Tensor<float>> *cacheResult = nullptr) {

			const size_t batchIndex = inputData.getShape()[0];


			/*	*/
			Tensor<float> layerResult = inputData;

			for (auto it = this->forwardSequence.begin(); it != this->forwardSequence.end(); it++) {
				Layer<T> *current = (*it);

				const bool junctionLayer = is_junction_layer(current);

				Tensor<float> batchTmp =
					Shape<unsigned int>({static_cast<unsigned int>(batchIndex)}).insert(1, current->getShape());

				/*	Compute each batch element.	*/
				for (size_t i = 0; i < batchSize; i++) {
					Tensor<float> batch = layerResult.getSubset({{static_cast<IndexType>(i)}});
					batch.reduce();

					Tensor<float> resultSubset = batchTmp.getSubset({static_cast<IndexType>(i)});

					/*	Perform layer on data.	*/
					resultSubset.assign((*current) << (const_cast<const Tensor<float> &>(batch)));
				}
				/*	Override the layer result with the batch.	*/
				layerResult = std::move(batchTmp);

				/*	validate result shape. */
				const auto shape =
					layerResult.getShape().getSubShape(layerResult.getShape().getNrDimensions() > 1 ? 1 : 0);
				if (shape != current->getShape()) {
					/*	*/
					std::cerr << "Invalid Shape: " << shape << " " << current->getShape() << " " << current->getName()
							  << std::endl;
				}

				/*	Assign result if memory provided.	*/
				if (cacheResult != nullptr) {
					(*cacheResult)[(*current).getName()] = layerResult;
				}
			}

			/*	*/
			result = std::move(layerResult);
		}

		void backPropagation(const Tensor<float> &error, std::map<std::string, Tensor<float>> &cacheResult) {

			Tensor<float> differental_error = error;

			Tensor<float> &layerResult = cacheResult[(*this->forwardSequence.rbegin())->getName()];

			/*	*/
			for (auto it = this->forwardSequence.rbegin(); it != this->forwardSequence.rend(); it++) {
				// std::cout << differental_error << std::endl << std::endl;
				Layer<T> *current = (*it);

				/*	*/
				differental_error = current->compute_derivative(static_cast<const Tensor<float> &>(layerResult));

				/*	Only apply if */
				Tensor<float> *train_variables = current->getTrainableWeights();
				if (train_variables) {
					this->optimizer->update_step(differental_error, *train_variables);
				}

				/*	*/
				layerResult = cacheResult[current->getName()];
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
					newName += static_cast<char>('0' + static_cast<char>(index));
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

				/*	*/
				if (current->getInputs().size() > 0) {
					current->build(current->getInputs()[0]->getShape());
				}

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
		std::map<std::string, Tensor<float>> history;

		/*	*/
		std::vector<Metric *> metrics;
		/*	TODO: impl	*/
		std::map<std::string, Object *> batchCache;
		std::map<std::string, Object *> backPropagationCache;

	  private: /*	Internal data.	*/
		std::list<Layer<DType> *> forwardSequence;
		size_t nr_weights;
		size_t trainableWeightSizeInBytes;
		size_t noneTrainableWeightSizeInBytes;
		std::string name;

	}; // namespace Ritsu
} // namespace Ritsu