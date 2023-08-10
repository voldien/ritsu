#pragma once
#include "../Random.h"
#include "../Tensor.h"
#include "../core/Shape.h"
#include <cstddef>
#include <omp.h>
#include <string>
#include <vector>

namespace Ritsu {

	template <typename T> class Layer {
	  public:
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");

		using IndexType = unsigned int;
		static constexpr size_t IndexTypeSize = sizeof(IndexType);
		using DType = T;
		const size_t DTypeSize = sizeof(DType);

	  public:
		Layer(const std::string &name) : name(name) { this->shape = Shape<IndexType>({1, 1}); }
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
		const Shape<IndexType> &getShape() const { return this->shape; }

		virtual void build(const std::vector<unsigned int> &dims) {}

		// virtual Tensor &operator()(const Tensor &tensor) { return tensor; }

		// Dtype

		// Weights trainable
		virtual Tensor *getTrainableWeights() { return nullptr; }

		// non-trainable.
		virtual Tensor *getVariables() { return nullptr; }

		// input
		virtual std::vector<Layer<T> *> getInputs() const { return {}; };

		// name

		// output
		virtual std::vector<Layer<T> *> getOutputs() const { return {}; }

		// trainable.

		const std::string &getName() const { return this->name; }
		void setName(const std::string &name) { this->name = name; }

		Layer<T> &operator()(Layer<T> &layer) {
			this->setInputs({&layer});
			layer.setOutputs({this});
			return *this;
		}

		virtual void setInputs(const std::vector<Layer<DType> *> &layers) {}
		virtual void setOutputs(const std::vector<Layer<DType> *> &layers) {}

		virtual Tensor compute_deriviate(const Tensor &tensor) { return tensor; }
		virtual Tensor &compute_deriviate(Tensor &tensor) const { return tensor; }

	  protected:
		Shape<IndexType> shape;

	  private:
		std::string name;
	};
} // namespace Ritsu