#pragma once
#include "../Object.h"
#include "../Tensor.h"
#include "../core/Shape.h"
#include <cstddef>
#include <omp.h>
#include <string>
#include <typeinfo>
#include <vector>

namespace Ritsu {

	/**
	 * @brief
	 *
	 * @tparam T
	 */
	// TODO add Object.
	template <typename T> class Layer {
	  public:
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");

		using IndexType = unsigned int;
		static constexpr size_t IndexTypeSize = sizeof(IndexType);
		using DType = T;
		const size_t DTypeSize = sizeof(DType);

	  public:
		Layer(const std::string &name) : name(name) { this->shape = std::move(Shape<IndexType>()); }
		virtual ~Layer() {}

		virtual Tensor operator<<(const Tensor &tensor) { return tensor; }

		virtual Tensor &operator<<(Tensor &tensor) { return tensor; }

		virtual Tensor operator>>(Tensor &tensor) { return tensor; }

		virtual Tensor &operator()(Tensor &tensor) { return tensor; }

		template <class U> auto &operator()(const U &layer) {
			this->getInputs()[0] = layer;
			return *this;
		}

		template <class U> auto &operator()(U &layer) {
			this->setInputs({&layer});
			layer.setOutputs({this});
			return *this;
		}

		const Shape<IndexType> &getShape() const { return this->shape; }

		virtual void build(const Shape<IndexType> &shape) {}

		// virtual Tensor &operator()(const Tensor &tensor) { return tensor; }

		// Dtype
		const std::type_info &getDType() const noexcept { return typeid(DType); }

		// Weights trainable
		virtual Tensor *getTrainableWeights() noexcept { return nullptr; }

		// non-trainable.
		virtual Tensor *getVariables() noexcept { return nullptr; }

		// input
		virtual std::vector<Layer<T> *> getInputs() const { return {}; };

		// output
		virtual std::vector<Layer<T> *> getOutputs() const { return {}; }

		// trainable.

		const std::string &getName() const noexcept { return this->name; }
		void setName(const std::string &name) noexcept { this->name = name; }

		Layer<T> &operator()(Layer<T> &layer) {
			this->connectLayers(layer);
			return *this;
		}

		virtual void connectLayers(Layer<T> &layer) {
			this->setInputs({&layer});
			layer.setOutputs({this});
		}

		/**
		 * @brief Override
		 *
		 * @param layers
		 */
		virtual void setInputs(const std::vector<Layer<DType> *> &layers) = 0;

		/**
		 * @brief Override the input
		 *
		 * @param layers
		 */
		virtual void setOutputs(const std::vector<Layer<DType> *> &layers) = 0;

		size_t getNrInputLayers() const noexcept { return this->getInputs().size(); }
		size_t getNrOutputLayers() const noexcept { return this->getOutputs().size(); }

		virtual Tensor compute_derivative(const Tensor &tensorLoss) = 0;
		virtual Tensor &compute_derivative(Tensor &tensorLoss) const = 0;

		void addInputLayers(const std::vector<Layer<DType> *> &layers) {
			/*	*/

			this->setInputs(layers);
		}
		void addOutputLayers(const std::vector<Layer<DType> *> &layers) {
			/*	*/
			this->setOutputs(layers);
		}

	  private:
	  protected:
		Shape<IndexType> shape;

		//		std::vector<Layer<DType> *> *input;
		// std::vector<Layer<DType> *> outputs;

	  private:
		std::string name;
	};
} // namespace Ritsu