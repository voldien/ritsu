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
	// TODO: add training bool
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
		virtual Tensor<float> &operator()(Tensor<float> &tensor, bool training = true) {
			return this->call(tensor, training);
		}

		virtual Tensor<float> operator()(const Tensor<float> &tensor, bool training = true) {
			return this->call(tensor, training);
		}

		virtual Tensor<float> &call(Tensor<float> &tensor, bool training) = 0;
		virtual Tensor<float> call(const Tensor<float> &tensor, bool training) = 0;

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

		virtual void build([[maybe_unused]] const Shape<IndexType> &buildShape) = 0;

		// Dtype
		const std::type_info &getDType() const noexcept { return typeid(DType); }

		// TODO: add array.
		// Weights trainable

		virtual std::optional<std::vector<Tensor<DType> *>> getTrainableWeights() noexcept { return {}; }

		// TODO: add array.
		// non-trainable.
		virtual std::vector<Tensor<DType> *> getVariables() noexcept { return {}; }

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