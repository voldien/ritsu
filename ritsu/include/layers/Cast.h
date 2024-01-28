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

		Tensor<float> &operator()(Tensor<float> &tensor) override {
			tensor = this->createCastTensor(tensor);
			return tensor;
		}

		Tensor<float> &operator<<(Tensor<float> &tensor) override {
			tensor = this->createCastTensor(tensor);
			return tensor;
		}

		Tensor<float> operator<<(const Tensor<float> &tensor) override { return this->createCastTensor(tensor); }

		template <class U> auto &operator()(U &layer) {

			this->setInputs({&layer});
			layer.setOutputs({this});

			this->build(layer.getShape());

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

		void build(const Shape<typename Layer<T>::IndexType> &shape) override { this->shape = shape; }

		std::vector<Layer<T> *> getInputs() const override { return {input}; }
		std::vector<Layer<T> *> getOutputs() const override { return outputs; }

		Tensor<float> compute_derivative(const Tensor<float> &tensor) override { return tensor; }
		Tensor<float> &compute_derivative(Tensor<float> &tensor) const override { return tensor; }

	  protected:
		// TODO: Fix refrence

		static Tensor<float> createCastTensor(const Tensor<float> &tensor) {
			Tensor<float> castTensor(tensor.getShape(), sizeof(T));

			return createCastTensorRef(castTensor);
		}

		static Tensor<float> &createCastTensorRef(Tensor<float> &tensor) {
			/*	*/
#pragma omp parallel for simd shared(tensor)
			for (size_t i = 0; i < tensor.getNrElements(); i++) {
				tensor.getValue<T>(i) = static_cast<T>(tensor.getValue<A>(i));
			}

			return tensor;
		}

	  private:
		Layer<T> *input;
		std::vector<Layer<T> *> outputs;
	};
} // namespace Ritsu