#pragma once
#include "../Object.h"
#include "../Tensor.h"
#include "../core/Shape.h"
#include <cstddef>
#include <cstdint>
#include <initializer_list>
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
	template <typename T> class Layer : public Object {
	  public:
		static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
					  "Must be a decimal type(float/double/half) or integer.");

		using IndexType = std::uint32_t;
		static constexpr unsigned int IndexTypeSize = sizeof(IndexType);
		using DType = T;
		const unsigned int DTypeSize = sizeof(DType);

	  public:
		Layer(const std::string &name) noexcept : Object(name) { this->shape = std::move(Shape<IndexType>()); }
		virtual ~Layer() noexcept {}

		/**
		 * @brief
		 */
		virtual Tensor<float> operator<<(const Tensor<float> &tensor) { return tensor; }

		/**
		 * @brief
		 */
		virtual Tensor<float> &operator<<(Tensor<float> &tensor) { return tensor; }

		/**
		 * @brief
		 */
		virtual Tensor<float> operator>>(Tensor<float> &tensor) { return tensor; }

		/**
		 * @brief
		 */
		virtual Tensor<float> &operator()(Tensor<float> &tensor) { return tensor; }

		/**
		 * @brief
		 */
		template <class... Arg> auto &operator()(const Arg &... layer) {
			this->getInputs() = {layer...};
			return *this;
		}

		/**
		 * @brief
		 */
		template <class... Arg> auto &operator()(Arg &... layer) {
			std::initializer_list<Layer *> list = {&layer...};
			this->connectLayers(list);
			return *this;
		}

		const Shape<IndexType> &getShape() const { return this->shape; }

		virtual void build(const Shape<IndexType> &shape) {}


		// Dtype
		const std::type_info &getDType() const noexcept { return typeid(DType); }

		// Weights trainable
		virtual Tensor<float> *getTrainableWeights() noexcept { return nullptr; }

		// non-trainable.
		virtual Tensor<float> *getVariables() noexcept { return nullptr; }

		// input
		virtual std::vector<Layer<T> *> getInputs() const { return {}; };

		// output
		virtual std::vector<Layer<T> *> getOutputs() const { return {}; }

		virtual void connectLayers(const std::initializer_list<Layer *> &list) {
			this->setInputs(list);

			/*	*/
			for (auto it = list.begin(); it != list.end(); it++) {
				(*it)->setOutputs({this});
			}
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

		virtual Tensor<float> compute_derivative(const Tensor<float> &tensorLoss) = 0;
		virtual Tensor<float> &compute_derivative(Tensor<float> &tensorLoss) const = 0;

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
	};
} // namespace Ritsu