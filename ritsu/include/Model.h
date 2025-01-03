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
#include "layers/Reshape.h"
#include "optimizer/Optimizer.h"
#include <cassert>
#include <chrono>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <list>
#include <map>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>

namespace Ritsu {

	template <typename T> void shuffle_data(Tensor<T> &data, const int axis = 0, const size_t seed = 0) {

		const size_t count = data.getShape()[0];
		const Shape<typename Tensor<T>::IndexType> shape = data.getShape().getSubShape(1);

		RandomUniform<float> randomGenerator(0, 1, seed);

		for (size_t i = 0; i < count / 2; i++) {

			const typename Tensor<T>::IndexType swap0 = std::floor(randomGenerator.rand() * count);
			const typename Tensor<T>::IndexType swap1 = std::floor(randomGenerator.rand() * count);

			auto A =
				std::move(data.getSubset(swap0 * shape.getNrElements(), (swap0 + 1) * shape.getNrElements(), shape));
			auto B =
				std::move(data.getSubset(swap1 * shape.getNrElements(), (swap1 + 1) * shape.getNrElements(), shape));

			std::swap(A, B);
		}
	}

	// Split
	template <typename T>
	std::tuple<Tensor<T>, Tensor<T>> split_dataset(const Tensor<T> &dataset, const float left_side = 0.5f,
												   bool shuffle = false, int seed = 0, const bool parent = true) {
		const size_t left_size_count = dataset.getShape()[0] * left_side;
		const size_t right_size_count = dataset.getShape()[0] - left_size_count;

		assert(left_size_count + right_size_count == dataset.getShape()[0]);

		if (left_side <= 0) {
			return {dataset, {}};
		}

		Shape<typename Tensor<T>::IndexType> leftShape = dataset.getShape();
		leftShape[0] = left_size_count;

		Shape<typename Tensor<T>::IndexType> rightShape = dataset.getShape();
		rightShape[0] = right_size_count;

		Tensor<T> leftSplit;
		Tensor<T> rightSplit;

		if (parent) {
			leftSplit = dataset.getSubset(0, leftShape.getNrElements(), leftShape);
			rightSplit = dataset.getSubset(leftShape.getNrElements(),
										   leftShape.getNrElements() + rightShape.getNrElements(), rightShape);
		} else {
			//		Tensor<T>(dataset.getRawData(), leftShape.getNrElements() * dataset.getElementSize(), leftShape );
		}

		if (shuffle) {
			shuffle_data(leftSplit, 0, seed);
			shuffle_data(rightSplit, 0, seed);
		}

		return {leftSplit, rightSplit};
	}

	/**
	 * @brief
	 *
	 * @tparam T
	 */
	template <typename T = float> class Model : public Object {
	  public:
		using DType = T;
		static constexpr const size_t DTypeSize = sizeof(DType);
		using IndexType = unsigned int;

		using History = std::map<std::string, Tensor<float>>;

		// TODO: define.
		using CallBack = void (*)(const Tensor<DType> &evaluated_pre_true, const Tensor<DType> &expected_pred,
								  Tensor<DType> &output_result);

	  public:
		Model(const std::vector<Layer<T> *> inputs, const std::vector<Layer<T> *> outputs,
			  const std::string &name = "model")
			: Object(name), inputs(inputs), outputs((outputs)) {
			// TODO build up memory.
			/*	TODO other optimization.	*/
			this->build(this->inputs, this->outputs);
		}
		virtual ~Model() = default;

		template <typename U, typename Y>
		History &fit(const size_t epochs, const Tensor<U> &inputData, const Tensor<Y> &expectedData,
					 const size_t batch_size = 1, const float validation_split = 0.0f, const bool shuffle = false,
					 const bool verbose = true, std::initializer_list<CallBack> callback = {}) {

			const size_t batch_shape_index = 0;

			/*	*/
			if (!this->is_built()) {
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
			/*	Verify Shapes and etc*/

			{}

			/*	*/
			auto [data_train, input_validation] = split_dataset<float>(inputData, 1 - validation_split, false, 0, true);
			auto [expected_train, expected_validation] =
				split_dataset<float>(expectedData, 1 - validation_split, false, 0, true);

			// TODO add support

			/*	Batch Shape Size.	*/
			Shape<Tensor<float>::IndexType> batchDataShape = inputData.getShape();
			batchDataShape[batch_shape_index] = batch_size;
			Shape<Tensor<float>::IndexType> batchExpectedShape = expectedData.getShape();
			batchExpectedShape[batch_shape_index] = batch_size;

			/*	*/
			const size_t batchDataElementSize = batchDataShape.getNrElements();
			const size_t batchExpectedElementSize = batchExpectedShape.getNrElements();

			// TODO add array support.
			Tensor<float> batchPredictedResult;
			/*	TODO: setup cache.	*/
			std::map<std::string, Tensor<float>> cachedResult;

			/*	Preallocate.	*/ // TODO:
			Tensor<float> loss_error = Tensor<float>(this->outputs[0]->getShape());
			Tensor<float> loss_deriv = Tensor<float>(this->outputs[0]->getShape());

			Tensor<float> timeSample({8});
			uint32_t timeIndex = 0;

			auto validation_metric = this->metrics;

			Time time;
			time.start();

			for (size_t nthEpoch = 0; nthEpoch < epochs; nthEpoch++) {

				if (verbose) {
					std::cout << "Epoch: " << nthEpoch << " / " << epochs << std::endl << std::flush;
				}

				/*	Shuffle.	*/
				if (shuffle) {
					const size_t shuffle_seed = rand();
					shuffle_data(data_train, 0, shuffle_seed);
					shuffle_data(expected_train, 0, shuffle_seed);
				}

				/*	Train pass.	*/
				for (size_t ibatch = 0; ibatch < nrTrainBatches; ibatch++) {

					/*	Extract subset of the data.	*/
					const Tensor<float> subsetBatchX =
						data_train.getSubset({{static_cast<unsigned int>(ibatch * batch_size),
											   static_cast<unsigned int>((ibatch + 1) * batch_size) - 1}});

					/*	*/
					const Tensor<float> subsetExpectedBatch =
						expected_train.getSubset({{static_cast<unsigned int>(ibatch * batch_size),
												   static_cast<unsigned int>((ibatch + 1) * batch_size) - 1}});

					/*	Compute network forward.	*/
					this->forwardPropgation(subsetBatchX, batchPredictedResult, batch_size, &cachedResult);

					/*	*/
					loss_error = this->lossFunction->computeLoss(subsetExpectedBatch, batchPredictedResult);
					loss_deriv = this->lossFunction->derivative(subsetExpectedBatch, batchPredictedResult);

					/*	*/
					debug_print_tensor(std::cout, loss_error, "loss: ");
					debug_print_tensor(std::cout, loss_deriv, "loss-derivative: ");

					/*	Apply metric update.	*/
					{
						this->lossmetric.update_state({&loss_error});

						assert(!std::isnan(this->lossmetric.result().getValue(0)));
						assert(!std::isinf(this->lossmetric.result().getValue(0)));

						for (size_t m_index = 0; m_index < this->metrics.size(); m_index++) {
							this->metrics[m_index]->update_state(subsetExpectedBatch, batchPredictedResult);
						}

						/*	Update history, using all metrics.	*/
						for (size_t m_index = 0; m_index < this->metrics.size(); m_index++) {
							/*	*/
							this->history[this->metrics[m_index]->getName()].concatenate(
								this->metrics[m_index]->result().template getValue<float>(0));
						}
						this->history[this->lossmetric.getName()].concatenate(this->lossmetric.result().getValue(0));
					}

					/*	*/
					this->backPropagation(loss_deriv, cachedResult, batch_size);

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

						// time.getElapsed<std::chrono::seconds>();

						std::chrono::duration_cast<std::chrono::seconds>(p);

						std::cout << "\33[2K\r" << "Batch: " << (ibatch + 1) << "/"
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
				for (size_t batch_index = 0; batch_index < nrValidationBatches && validation_split > 0; batch_index++) {

					/*	Extract subset of the data.	*/
					const size_t baseBatch = batch_index;

					/*	Extract subset of the data.	*/
					const Tensor<float> subsetBatchX =
						input_validation.getSubset({{static_cast<unsigned int>(baseBatch * batch_size),
													 static_cast<unsigned int>((baseBatch + 1) * batch_size) - 1}});

					/*	*/
					const Tensor<float> subsetExpectedBatch =
						expected_validation.getSubset({{static_cast<unsigned int>(baseBatch * batch_size),
														static_cast<unsigned int>((baseBatch + 1) * batch_size) - 1}});

					/*	Compute network forward.	*/
					this->forwardPropgation(subsetBatchX, batchPredictedResult, batch_size, nullptr);

					/*	*/
					loss_error = std::move(this->lossFunction->computeLoss(subsetExpectedBatch, batchPredictedResult));
					
					/*	Apply metric update.	*/
					{
						this->lossmetric.update_state({&loss_error});

						assert(!std::isnan(this->lossmetric.result().getValue(0)));
						assert(!std::isinf(this->lossmetric.result().getValue(0)));

						for (size_t m_index = 0; m_index < this->metrics.size(); m_index++) {
							this->metrics[m_index]->update_state(subsetExpectedBatch, batchPredictedResult);
						}

						/*	Update history, using all metrics.	*/
						for (size_t m_index = 0; m_index < this->metrics.size(); m_index++) {
							/*	*/
							this->history[this->metrics[m_index]->getName()].concatenate(
								this->metrics[m_index]->result().template getValue<float>(0));
						}
						this->history[this->lossmetric.getName()].concatenate(this->lossmetric.result().getValue(0));
					}
				}

				/*	Update history, using all metrics.	*/
				for (size_t m_index = 0; m_index < this->metrics.size(); m_index++) {
					/*	*/
					// this->history[this->metrics[m_index]->getName()].concatenate(
					//	this->metrics[m_index]->result().template getValue<float>(0));
				}

				if (verbose) {
					std::cout << std::endl << std::flush;
				}
			}

			return this->history;
		}

		template <typename U, typename Y>
		Tensor<Y> predict(const Tensor<U> &inputTensor, const size_t batch = 1, const bool verbose = false) {

			Tensor<float> result;

			Time time;

			time.start();

			this->forwardPropgation(inputTensor, result, batch);

			time.getElapsed<float>();

			return result;
		}

		void compile(Optimizer<T> *optimizer, const Loss<T> &loss, const std::vector<Metric *> &compile_metrics = {}) {

			if (optimizer == nullptr) {
			}

			this->optimizer = optimizer;
			this->lossFunction = &loss;

			this->lossmetric = MetricMean("loss");
			/*	Compile metrics.	*/
			this->metrics = compile_metrics;

			/*	*/
			for (size_t m_index = 0; m_index < this->metrics.size(); m_index++) {
				this->history[this->metrics[m_index]->getName()] = Tensor<float>({1});
			}
			this->history[this->lossmetric.getName()] = Tensor<float>({1});

			/*	Cache result tensors.*/
			// TODO: add support for caching tensors.
			std::vector<Tensor<float> *> tensors;
			for (auto it = this->forwardSequence.begin(); it != this->forwardSequence.end(); it++) {
			}

			/*	build optimizer.	*/
			// TODO: add support for caching build.
			// this->optimizer->build({});
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

				_summary << current->getName() << '\t' << " " << current->getShape() << '\t';

				/*	*/
				for (size_t i = 0; i < current->getInputs().size(); i++) {

					if (i == 0) {
						_summary << "<-- [ ";
					}
					_summary << current->getInputs()[i]->getName() << " ";
					if (i == current->getInputs().size() - 1) {
						_summary << "]" << " " << current->getDType().name();
					}
				}

				_summary << std::endl;
			}

			const size_t train_in_bytes = static_cast<size_t>(this->trainableWeightSizeInBytes / 1024.0f);
			const size_t none_train_in_bytes = static_cast<size_t>(this->noneTrainableWeightSizeInBytes / 1024.0f);

			/*	Summary of number of parameters and size.	*/
			_summary << "number of weights: " << std::to_string(this->nr_weights) << std::endl;
			_summary << "Trainable in Bytes: " << std::to_string(train_in_bytes) << " KB" << std::endl;
			_summary << "None-Trainable in Bytes: " << std::to_string(none_train_in_bytes) << " KB" << std::endl;
			_summary << "Loss Function: " << this->lossFunction->getName() << std::endl;
			_summary << "Optimizer: " << this->optimizer->getName() << std::endl;
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
		template <typename U, typename Y>
		void forwardPropgation(const Tensor<U> &inputData, Tensor<Y> &result, const size_t batchSize,
							   std::map<std::string, Tensor<float>> *cacheResult = nullptr) {

			const size_t batchIndex = inputData.getShape()[0];

			/*	*/
			Tensor<float> layerResult = inputData;
			const bool is_training = cacheResult != nullptr;

			if (cacheResult) {
				(*cacheResult)["inputData"] = inputData;
			}

			for (auto it = this->forwardSequence.begin(); it != this->forwardSequence.end(); it++) {
				Layer<T> *current = (*it);

				const bool junctionLayer = is_junction_layer(current);

				Tensor<float> batchTmp =
					Shape<unsigned int>({static_cast<unsigned int>(batchIndex)}).insert(1, current->getShape());

				/*	Compute each batch element.	*/
				for (size_t batch_index = 0; batch_index < batchSize; batch_index++) {
					/*	*/
					Tensor<float> prevBatch = layerResult.getSubset({{static_cast<IndexType>(batch_index)}});
					prevBatch.reduce();

					// std::cerr << prevBatch << std::endl << std::endl;
					Tensor<float> resultSubset = batchTmp.getSubset({static_cast<IndexType>(batch_index)});

					/*	Perform layer on data.	*/
					Tensor<float> result = current->call(const_cast<const Tensor<float> &>(prevBatch), is_training);
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
					(*cacheResult)[(*current).getName()].transpose();
				}
			}

			/*	*/
			result = std::move(layerResult);
		}

		void backPropagation(const Tensor<float> &error, std::map<std::string, Tensor<float>> &cacheResult,
							 const size_t batchSize) {

			/*	*/
			Tensor<float> &current_layer_q = cacheResult[(*this->forwardSequence.rbegin())->getName()];
			Tensor<float> &previous_layer_q = cacheResult[(*this->forwardSequence.rbegin()++)->getName()];

			const float batch_inverse = 1.0f / static_cast<float>(batchSize);

			Tensor<float> differental_z_error = error.transpose();

			/*	Duplicate the loss to match the batch size.	*/
			for (unsigned int i = 0; i < batchSize - 1; i++) {
				differental_z_error.concatenate(error.transpose());
			}

			// TODO:
			Shape<IndexType> diffShape;
			diffShape.insert(0, {(IndexType)batchSize});
			diffShape.insert(1, error.getShape().getSubShape(1));

			differental_z_error.reshape(diffShape);
			differental_z_error.transpose();

			Tensor<float> prev_layer_deriv = differental_z_error;

			/*	*/
			Layer<T> *current = nullptr;
			Layer<T> *prev = nullptr;
			for (auto it = this->forwardSequence.rbegin(); it != this->forwardSequence.rend(); it++) {

				/*	*/
				current = (*it);
				prev = *(std::next(it));

				/*	Finished.	*/
				if (is_input_layer(current)) {
					return;
				}

				/*	*/
				if (cacheResult.find(current->getName()) != cacheResult.end()) {
					current_layer_q = cacheResult[current->getName()];
				} else {
					current_layer_q = cacheResult["inputData"];
				}
				if (std::next(it) != this->forwardSequence.rend() &&
					cacheResult.find(prev->getName()) != cacheResult.end()) {
					previous_layer_q = cacheResult[prev->getName()];
				}

				/*	Extract if any trainable.	 */
				std::optional<std::vector<Tensor<DType> *>> optional_train_variables = current->getTrainableWeights();

				/*	*/
				debug_print_layer<DType>(std::cout, *current);

				debug_print_tensor(std::cout, differental_z_error, "dZ");

				/*	*/
				if (optional_train_variables.has_value() && !optional_train_variables.value().empty()) {
					std::vector<Tensor<DType> *> train_variables = optional_train_variables.value();

					/*	*/
					Tensor<float> differental_q_error =
						current->compute_derivative(static_cast<const Tensor<float> &>(differental_z_error));
					prev_layer_deriv = differental_q_error;

					debug_print_tensor(std::cout, previous_layer_q, "Q-1");

					debug_print_tensor(std::cout, differental_q_error, "DQ");

					/*	*/
					for (size_t i_var = 0; i_var < train_variables.size(); i_var++) {
						Tensor<DType> *variable = train_variables[i_var];

						Tensor<float> gradient =
							current->compute_gradient(i_var, differental_z_error, previous_layer_q);

						debug_print_tensor(std::cout, gradient, "Gradient");

						/*	*/
						gradient *= batch_inverse;

						this->optimizer->update_step(reinterpret_cast<Tensor<T> &>(gradient),
													 reinterpret_cast<Tensor<T> &>(*variable));
					}

					/*	*/
					differental_z_error = prev_layer_deriv;

				} else {

					/*	Update delta.	*/
					if (typeid(*current) == typeid(Reshape)) {
						differental_z_error.reshape(current->getInputs().at(0)->getShape());
					} else {
						Tensor<float> z_derv =
							current->compute_derivative(static_cast<const Tensor<float> &>(current_layer_q));
						differental_z_error = z_derv.dot(prev_layer_deriv);
					}
					prev_layer_deriv = differental_z_error;
					/*	*/
					debug_print_tensor_layer<T>(std::cout, *current,
												reinterpret_cast<Tensor<T> &>(differental_z_error));
				}
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

				if (this->is_junction_layer(current)) {
					throw NotSupportedException("Junction Layer not Supported");
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

				std::vector<Layer<T> *> layers = current->getOutputs();
				/*	*/
				if (!layers.empty()) {
					current = layers[0];
				} else {
					current = nullptr;
				}
			}
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

					index++; /*	Keep track of nth version of the name*/
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

		bool is_built() const noexcept {
			return layers.size() > 0 && optimizer && lossFunction && inputs.size() > 0 && outputs.size() > 0 &&
				   forwardSequence.size() > 0;
		}

		bool is_junction_layer(const Layer<DType> *layer) const noexcept { return layer->getInputs().size() > 1; }

		bool is_input_layer(const Layer<T> *layer) const noexcept {
			for (size_t i = 0; i < inputs.size(); i++) {
				if (layer == inputs[i]) {
					return true;
				}
			}
			return false;
		}

	  protected:
		/*	*/
		std::vector<Layer<T> *> inputs;
		std::vector<Layer<T> *> outputs;

		/*	*/
		Optimizer<T> *optimizer;
		const Loss<T> *lossFunction;

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
		size_t nr_weights{};
		size_t trainableWeightSizeInBytes{};
		size_t noneTrainableWeightSizeInBytes{};

	}; // namespace Ritsu
} // namespace Ritsu