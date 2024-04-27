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
#include "Math.h"
#include "Metric.h"
#include "Object.h"
#include "RitsuDebug.h"
#include "RitsuDef.h"
#include "Tensor.h"
#include "core/Shape.h"
#include "core/Time.h"
#include "layers/Layer.h"
#include "optimizer/Optimizer.h"
#include <cassert>
#include <chrono>
#include <cstddef>
#include <exception>
#include <iostream>
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

		using History = std::map<std::string, Tensor<float>>;

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
		History &fit(const size_t epochs, const Tensor<float> &inputData, const Tensor<float> &expectedData,
					 const size_t batch_size = 1, const float validation_split = 0.0f, const bool shuffle = false,
					 const bool verbose = true) {

			const size_t batch_shape_index = 0;

			/*	*/
			if (!this->is_build()) {
				/*	Invalid state.	*/
				throw RuntimeException("Model not built and compiled");
			}

			/*	Number of batches in dataset.	*/
			const size_t batchXIndex = inputData.getShape()[batch_shape_index];
			const size_t batchYIndex = expectedData.getShape()[batch_shape_index];

			/*	*/
			size_t nrTrainBatches = std::floor(static_cast<float>(batchXIndex) / static_cast<float>(batch_size));
			const size_t nrTrainExpectedBatches =
				std::floor(static_cast<float>(batchYIndex) / static_cast<float>(batch_size));

			nrTrainBatches = Math::min<size_t>(nrTrainBatches, nrTrainExpectedBatches);

			const size_t nrValidationBatches = std::floor(nrTrainBatches * validation_split);
			nrTrainBatches -= nrValidationBatches;

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
			Tensor<float> loss_error;
			Tensor<float> loss_deriv;

			for (size_t nthEpoch = 0; nthEpoch < epochs; nthEpoch++) {
				if (verbose) {
					std::cout << "Epoch: " << nthEpoch << " / " << epochs << std::endl << std::flush;
				}

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

					// std::cout << std::endl << subsetExpectedBatch << std::endl;

					loss_error = this->lossFunction->computeLoss(subsetExpectedBatch, batchResult);
					loss_deriv = this->lossFunction->derivative(subsetExpectedBatch, batchResult);

					debug_print_tensor(std::cout, loss_error, "loss");
					debug_print_tensor(std::cout, loss_deriv, "loss-derivative");

					/*	Apply metric update.	*/
					{
						this->lossmetric.update_state({&loss_error});

						assert(!std::isnan(this->lossmetric.result().getValue(0)));
						assert(!std::isinf(this->lossmetric.result().getValue(0)));

						for (size_t m_index = 0; m_index < this->metrics.size(); m_index++) {
							this->metrics[m_index]->update_state(subsetExpectedBatch, batchResult);
						}

						/*	Update history, using all metrics.	*/
						for (size_t m_index = 0; m_index < this->metrics.size(); m_index++) {
							/*	*/
							this->history[this->metrics[m_index]->getName()].concatenate(
								this->metrics[m_index]->result().template getValue<float>(0));
						}
						this->history[this->lossmetric.getName()].concatenate(this->lossmetric.result().getValue(0));

						// std::cout << std::endl << std::endl << "loss: " << loss_error << std::endl << std::endl;
						// std::cout << std::endl << "loss derv: " << loss_deriv << std::endl << std::endl;
					}

					/*	*/
					this->backPropagation(loss_deriv, cachedResult);

					/*	*/
					if (verbose) {
						this->print_status(std::cout);
					}

					if (verbose) {
						/*	*/
						timeSample.getValue<float>(timeIndex) = time.deltaTime<float>();
						timeIndex = (timeIndex + 1) % timeSample.getNrElements();
						const float averageTime = Tensor<float>::mean<float>(timeSample);

						const size_t nrBatchPerSecond = 1.0f / averageTime;
						const float expectedEpochTime = (nrTrainBatches - ibatch) * averageTime;

						using fsec = duration<float>;
						auto p = round<nanoseconds>(fsec{expectedEpochTime});

						std::chrono::duration_cast<std::chrono::seconds>(p);

						std::cout << "\33[2K\r"
								  << "Batch: " << ibatch << "/"
								  << nrTrainBatches // << " " << nrBatchPerSecond << "batch/Sec"
								  << " ETA: " << std::chrono::duration_cast<std::chrono::seconds>(p).count()
								  << " - lr: " << this->optimizer->getLearningRate();
						std::cout << " loss: " << this->lossmetric.result().getValue(0);

						for (size_t m_index = 0; m_index < this->metrics.size(); m_index++) {
							std::cout << " - " << this->metrics[m_index]->getName() << ": "
									  << Tensor<float>::mean<DType>(this->metrics[m_index]->result());
						}
					}
					if (verbose) {
						std::cout << std::flush;
					}

					time.update();
				}

				/*	Validation Pass. Only compute, no backpropgation.	*/
				for (size_t ibatch = 0; ibatch < nrValidationBatches; ibatch++) {

					/*	Extract subset of the data.	*/
					const size_t baseBatch = (nrTrainBatches + ibatch);
					/*	Extract subset of the data.	*/
					const Tensor<float> subsetBatchX =
						inputData.getSubset({{static_cast<unsigned int>(baseBatch * batch_size),
											  static_cast<unsigned int>((baseBatch + 1) * batch_size) - 1}});

					/*	*/
					const Tensor<float> subsetExpectedBatch =
						expectedData.getSubset({{static_cast<unsigned int>(baseBatch * batch_size),
												 static_cast<unsigned int>((baseBatch + 1) * batch_size) - 1}});

					/*	Compute network forward.	*/
					this->forwardPropgation(subsetBatchX, batchResult, batch_size, nullptr);
					/*	*/
					loss_error = std::move(this->lossFunction->computeLoss(subsetExpectedBatch, batchResult));
				}

				/*	Update history, using all metrics.	*/
				for (size_t m_index = 0; m_index < this->metrics.size(); m_index++) {
					/*	*/
					this->history[this->metrics[m_index]->getName()].concatenate(
						this->metrics[m_index]->result().template getValue<float>(0));
				}

				if (verbose) {
					std::cout << std::endl << std::flush;
				}
			}

			return this->history;
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

		void compile(Optimizer<T> *optimizer, const Loss &loss, const std::vector<Metric *> &compile_metrics = {}) {
			this->optimizer = optimizer;
			this->lossFunction = &loss;

			this->lossmetric = MetricMean("loss");
			/*	Compile metrics.	*/
			this->metrics = compile_metrics;
			// TODO verify the objects.

			/*	*/
			for (size_t m_index = 0; m_index < this->metrics.size(); m_index++) {
				this->history[this->metrics[m_index]->getName()] = Tensor<float>({1});
			}
			this->history[this->lossmetric.getName()] = Tensor<float>({1});

			/*	*/ // Cache result tensors.
			std::vector<Tensor<float> *> tensors;
			for (auto it = this->forwardSequence.begin(); it != this->forwardSequence.end(); it++) {
				//(*it)->
			}
			/*	build optimizer.	*/
			// this->optimizer->build({Shape<IndexType>({32, 32, 32})});
		}

		Layer<DType> *getLayer(const std::string &name) { return this->layers[name]; }

		virtual std::string summary() const {
			std::stringstream _summary;

			// TODO: add support for aligned summary.
			size_t maxSpaceForTab = 0;
			for (auto it = this->forwardSequence.begin(); it != this->forwardSequence.end(); it++) {
				const Layer<T> *current = (*it);
				maxSpaceForTab = Math::max<size_t>(current->getName().size(), maxSpaceForTab);
			}

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
						_summary << "]"
								 << " " << current->getDType().name();
					}
				}

				_summary << std::endl;
			}

			/*	Summary of number of parameters and size.	*/
			_summary << "number of weights: " << std::to_string(this->nr_weights) << std::endl;
			_summary << "Trainable in Bytes: "
					 << std::to_string(static_cast<size_t>(this->trainableWeightSizeInBytes / 1024.0f)) << " KB"
					 << std::endl;
			_summary << "None-Trainable in Bytes: "
					 << std::to_string(static_cast<size_t>(this->noneTrainableWeightSizeInBytes / 1024.0f)) << " KB";
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
			const bool training = cacheResult != nullptr;

			for (auto it = this->forwardSequence.begin(); it != this->forwardSequence.end(); it++) {
				Layer<T> *current = (*it);

				const bool junctionLayer = is_junction_layer(current);

				Tensor<float> batchTmp =
					Shape<unsigned int>({static_cast<unsigned int>(batchIndex)}).insert(1, current->getShape());

				/*	Compute each batch element.	*/
				for (size_t i = 0; i < batchSize; i++) {
					/*	*/
					Tensor<float> prevBatch = layerResult.getSubset({{static_cast<IndexType>(i)}});
					prevBatch.reduce();

					// std::cerr << prevBatch << std::endl << std::endl;
					Tensor<float> resultSubset = batchTmp.getSubset({static_cast<IndexType>(i)});

					/*	Perform layer on data.	*/
					Tensor<float> result = current->call(const_cast<const Tensor<float> &>(prevBatch), training);
					resultSubset.assign(result);
				}
				/*	Override the layer result with the batch.	*/

				layerResult = std::move(batchTmp);

				/*	*/
				debug_print_tensor_layer<T>(std::cout, *current, reinterpret_cast<Tensor<T> &>(layerResult));

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

			Tensor<float> &current_layer_z = cacheResult[(*this->forwardSequence.rbegin())->getName()];
			const IndexType batch_size = current_layer_z.getShape()[0];
			const float batch_inverse = 1.0f / batch_size;

			Tensor<float> differental_error = error;

			/*	TODO: determine.	*/
			for (int i = 0; i < batch_size - 1; i++) {
				differental_error.concatenate(error);
			}
			// TODO:
			Shape<IndexType> diffShape;
			diffShape.insert(0, {(IndexType)batch_size});
			diffShape.insert(1, error.getShape().getSubShape(1));

			differental_error.reshape(diffShape);

			/*	*/
			for (auto it = this->forwardSequence.rbegin(); it != this->forwardSequence.rend(); it++) {

				Layer<T> *current = (*it);

				auto tmpIt = it;
				tmpIt++;
				Layer<T> *prev = (*(tmpIt));

				current_layer_z = cacheResult[current->getName()];

				/*	Only apply if */
				std::optional<std::vector<Tensor<DType> *>> optional_train_variables = current->getTrainableWeights();

				/*	*/
				debug_print_layer<DType>(std::cout, *current);

				if (optional_train_variables.has_value() && !optional_train_variables.value().empty()) {
					std::vector<Tensor<DType> *> train_variables = optional_train_variables.value();

					Tensor<float> layer_ahead_z = cacheResult[prev->getName()];
					Tensor<float> partial_error = differental_error;

					/*	Add batch dim.	*/
					if (partial_error.getShape().getNrDimensions() == 1) {
						partial_error.reshape({1, partial_error.getShape().getAxisDimensions(0)});
					}

					Tensor<float> weightGradient = layer_ahead_z.transpose().dot(partial_error);

					for (size_t i_var = 0; i_var < train_variables.size(); i_var++) {
						Tensor<DType> *variable = train_variables[i_var];

						if (variable) {
							this->optimizer->update_step(reinterpret_cast<Tensor<T> &>(weightGradient.transpose()),
														 reinterpret_cast<Tensor<T> &>(*variable));
						}
					}

					/*	*/
					differental_error =
						current->compute_derivative(static_cast<const Tensor<float> &>(current_layer_z));

					// this->optimizer->update_step(reinterpret_cast<Tensor<T> &>(weightGradient.transpose()),
					//							 reinterpret_cast<Tensor<T> &>(*train_variables));

					// partial_error = Tensor<float>::mean(partial_error, 1);
					// if (partial_error.getShape().getNrDimensions() == 1) {
					//	partial_error.reshape({1, partial_error.getShape().getAxisDimensions(0)});
					// }
					//					this->optimizer->update_step(reinterpret_cast<Tensor<T> &>(partial_error),
					//												 reinterpret_cast<Tensor<T> &>(*_variables));
				} else {

					/*	Update delta.	*/
					Tensor<float> z_derv =
						current->compute_derivative(static_cast<const Tensor<float> &>(current_layer_z));
					differental_error = differental_error.dot(z_derv);
				}
				/*	*/
				debug_print_layer<DType>(std::cout, *current);
			}
		}

		virtual void build(std::vector<Layer<T> *> inputs, std::vector<Layer<T> *> outputs) { /*	*/

			/*	*/
			this->forwardSequence.clear();

			// Iterate through each and extract number of trainable variables.
			this->build_sequence(inputs, outputs);
			this->init_unique_name();
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

				this->forwardSequence.push_back(current);

				// TODO add support.
				if (this->is_junction_layer(current)) {
				}

				std::optional<std::vector<Tensor<DType> *>> optional_train_variables = current->getTrainableWeights();

				if (optional_train_variables.has_value()) {
					std::vector<Tensor<DType> *> train_variables = optional_train_variables.value();

					if (!train_variables.empty()) {
						for (size_t i = 0; i < train_variables.size(); i++) {
							Tensor<DType> *variable = train_variables[i];
							this->nr_weights += variable->getNrElements();
							this->trainableWeightSizeInBytes += variable->getDatSize();
						}
					}
				}

				// if (!train_variables.empty()) {
				//	for (size_t i = 0; i < current->getVariables().size(); i++) {
				//		Tensor<DType> *variable = current->getVariables()[i];
				//		this->noneTrainableWeightSizeInBytes += variable->getDatSize();
				//	}
				//	//
				//}
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

		void init_unique_name() {

			/*	Extract all layer and rename to make them all unique.	*/
			for (auto it = this->forwardSequence.rbegin(); it != this->forwardSequence.rend(); it++) {

				/*	*/
				const std::string &name = (*it)->getName();
				std::string newName = (*it)->getName();

				int index = 0;
				while (this->layers.find(newName) != this->layers.end()) {

					newName = name + "_" + (*it)->getDType().name();
					newName += static_cast<char>('0' + static_cast<char>(index));

					index++; /*	*/
				}

				(*it)->setName(newName);
				this->layers[newName] = (*it);
			}
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
		const Loss *lossFunction;

		/*	*/
		std::map<std::string, Layer<T> *> layers;

		/*	*/
		std::map<std::string, Tensor<float>> history;

		/*	*/
		std::vector<Metric *> metrics;
		MetricMean lossmetric;
		/*	TODO: impl	*/
		std::map<std::string, Object *> batchCache;
		std::map<std::string, Object *> backPropagationCache;

	  private: /*	Internal data.	*/
		std::list<Layer<DType> *> forwardSequence;
		size_t nr_weights;
		size_t trainableWeightSizeInBytes;
		size_t noneTrainableWeightSizeInBytes;

	}; // namespace Ritsu
} // namespace Ritsu