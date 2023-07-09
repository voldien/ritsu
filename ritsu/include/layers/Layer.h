#pragma once
#include "../Random.h"
#include "../Tensor.h"
#include <cstddef>
#include <omp.h>
#include <string>
#include <vector>

namespace Ritsu {

	template <typename T> class Layer {

	  public:
		using DType = T;
		const size_t DTypeSize = sizeof(DType);
		Layer(const std::string &name) : name(name) {}
		virtual ~Layer() {}

		virtual Tensor operator<<(const Tensor &tensor) { return tensor; }

		virtual Tensor &operator<<(Tensor &tensor) { return tensor; }

		virtual Tensor operator>>(Tensor &tensor) { return tensor; }

		virtual Tensor &operator()(Tensor &tensor) { return tensor; }

		template <class U> auto &operator()(const U &layer) {
			this->getInputs()[0] = layer;
			return *this;
		}

		// TODO add shape.
		const std::vector<unsigned int> &getNrDimension() const { return this->shape; }

		virtual void build(const std::vector<unsigned int> &dims) {}

		// virtual Tensor &operator()(const Tensor &tensor) { return tensor; }

		// Dtype

		// Weights trainable
		virtual Tensor *getTrainableWeights() { return nullptr; }
		virtual Tensor *getVariables() { return nullptr; }

		// non-trainable.

		// input
		virtual std::vector<Layer<T> *> getInputs() const { return {}; };

		// name

		// output
		virtual std::vector<Layer<T> *> getOutputs() const { return {}; }

		// trainable.

		const std::string &getName() const { return this->name; }

		Layer<T> &operator()(Layer<T> &layer) {
			this->setInputs({&layer});
			layer.setOutputs({this});
			return *this;
		}

		virtual void setInputs(const std::vector<Layer<DType> *> &layers) {}
		virtual void setOutputs(const std::vector<Layer<DType> *> &layers) {}

	  protected:
		std::vector<unsigned int> shape = {1, 1};

	  private:
		std::string name;
	};
} // namespace Ritsu