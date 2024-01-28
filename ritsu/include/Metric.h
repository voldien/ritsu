#pragma once
#include "Object.h"
#include "Tensor.h"
#include <cstdarg>
#include <string>

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class Metric : public Object {
	  public:
		Metric(const std::string &m_name) : Object(m_name) {}

		virtual void update_state(const Tensor<float> &args...) = 0;

		virtual void reset_state() = 0;

		virtual const Tensor<float> &result() const noexcept = 0;

	  protected:
	};

	/**
	 * @brief
	 *
	 */
	class MetricAccuracy : public Metric {
	  public:
		MetricAccuracy(const std::string &name = "accuracy") : Metric(name) { this->reset_state(); }

		void update_state(const Tensor<float> &tensor...) override { /*	*/
			va_list args;
			va_start(args, tensor);

			const Tensor<float> *refA = &tensor;
			const Tensor<float> *refB = va_arg(args, const Tensor<float> *);

			size_t correct = 0;
			for (size_t i = 0; i < refA->getNrElements(); i++) {
				if (refA->getValue<float>(i) == refB->getValue<float>(i)) {
					correct++;
				}
			}

			this->m_result.getValue<float>(0) = (float)correct / (float)refA->getNrElements();
			va_end(args);
		}

		void reset_state() override { m_result = Tensor<float>({1}, sizeof(float)); }

		const Tensor<float> &result() const noexcept override { return this->m_result; }

	  private:
		Tensor<float> m_result;
	};

	/**
	 * @brief
	 *
	 */
	class MetricMean : public Metric {
	  public:
		MetricMean(const std::string &name = "mean") : Metric(name) { this->reset_state(); }

		void update_state(const Tensor<float> &tensor...) override {

			va_list args;
			va_start(args, tensor);

			const Tensor<float> *refA = &tensor;
			m_result.getValue<float>(0) = Tensor<float>::mean<float>(*refA);

			va_end(args);
		}

		void reset_state() override {
			this->m_result = Tensor<float>({1}, sizeof(float));
			this->m_result.getValue<float>(0) = 1;
		}

		const Tensor<float> &result() const noexcept override { return this->m_result; }

	  private:
		Tensor<float> m_result;
	};

} // namespace Ritsu