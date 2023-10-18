#pragma once
#include "Layer.h"
#include "Tensor.h"
#include <cstdint>

namespace Ritsu {

	/**
	 * @brief
	 *
	 * @tparam T
	 */
	template <typename A, typename T> class Cast : public Layer<T> {
	  public:
		Cast(const std::string &name = "cast") : Layer<T>(name){};

		Tensor &operator()(Tensor &tensor) override {
			tensor = this->createCastTensor(tensor);
			return tensor;
		}

		Tensor &operator<<(Tensor &tensor) override {
			tensor = this->createCastTensor(tensor);
			return tensor;
		}

		Tensor operator<<(const Tensor &tensor) override { return this->createCastTensor(tensor); }

		template <class U> auto &operator()(U &layer) {

			this->setInputs({&layer});
			layer.setOutputs({this});

			return *this;
		}

		void setInputs(const std::vector<Layer<T> *> &layers) override {
			this->input = layers[0];

			this->shape = this->input->getShape();
		}

		void setOutputs(const std::vector<Layer<T> *> &layers) override {
			/*	Set input layer */
			this->outputs = layers;
		}

		std::vector<Layer<T> *> getInputs() const override { return {input}; }
		std::vector<Layer<T> *> getOutputs() const override { return outputs; }

		Tensor compute_derivative(const Tensor &tensor) override { return tensor; }
		Tensor &compute_derivative(Tensor &tensor) const override { return tensor; }

	  protected:
		// TODO fix performance memory with reference..
		Tensor createCastTensor(const Tensor &tensor) {
			Tensor castTensor(tensor.getShape(), sizeof(T));

			for (size_t i = 0; i < tensor.getNrElements(); i++) {
				castTensor.getValue<T>(i) = static_cast<T>(tensor.getValue<A>(i));
			}

			return castTensor;
		}

	  private:
		Layer<T> *input;
		std::vector<Layer<T> *> outputs;
	};
} // namespace Ritsu