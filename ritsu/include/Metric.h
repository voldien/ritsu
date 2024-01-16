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

		virtual void update_state(const Tensor &args...) = 0;

		virtual void reset_state() = 0;

		virtual const Tensor &result() const noexcept = 0;

	  protected:
	};

	/**
	 * @brief
	 *
	 */
	class MetricAccuracy : public Metric {
	  public:
		MetricAccuracy(const std::string &name = "accuracy") : Metric(name) { this->reset_state(); }

		void update_state(const Tensor &tensor...) override { /*	*/
			va_list args;
			va_start(args, tensor);

			const Tensor *refA = &tensor;
			const Tensor *refB = va_arg(args, const Tensor *);

			size_t correct = 0;
			for (size_t i = 0; i < refA->getNrElements(); i++) {
				if (refA->getValue<float>(i) == refB->getValue<float>(i)) {
					correct++;
				}
			}

			this->m_result.getValue<float>(0) = (float)correct / (float)refA->getNrElements();
			va_end(args);
		}

		void reset_state() override { m_result = Tensor({1}, sizeof(float)); }

		const Tensor &result() const noexcept override { return this->m_result; }

	  private:
		Tensor m_result;
	};

	/**
	 * @brief
	 *
	 */
	class MetricMean : public Metric {
	  public:
		MetricMean(const std::string &name = "mean") : Metric(name) { this->reset_state(); }

		void update_state(const Tensor &tensor...) override {

			va_list args;
			va_start(args, tensor);

			const Tensor *refA = &tensor;
			m_result.getValue<float>(0) = Tensor::mean<float>(*refA);

			va_end(args);
		}

		void reset_state() override {
			this->m_result = Tensor({1}, sizeof(float));
			this->m_result.getValue<float>(0) = 1;
		}

		const Tensor &result() const noexcept override { return this->m_result; }

	  private:
		Tensor m_result;
	};

} // namespace Ritsu