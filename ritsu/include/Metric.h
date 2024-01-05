#pragma once
#include "Object.h"
#include "Tensor.h"
#include <string>

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class Metric : public Object {
	  public:
		Metric(const std::string &m_name) : Object(m_name) {}

		virtual void update_state(const Tensor &tensor) = 0;

		virtual void reset_state() = 0;

		virtual const Tensor &result() const = 0;

	  private:
	};

	/**
	 * @brief
	 *
	 */
	class MetricAccuracy : public Metric {
	  public:
		MetricAccuracy(const std::string &name = "accuracy") : Metric(name) { this->reset_state(); }

		void update_state(const Tensor &tensor) override { /*	*/
		}

		void reset_state() override { m_result = Tensor({1}); }

		const Tensor &result() const override { return this->m_result; }

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

		void update_state(const Tensor &tensor) override { m_result.getValue<float>(0) = Tensor::mean<float>(tensor); }

		void reset_state() override { m_result = Tensor({1}); }

		const Tensor &result() const override { return this->m_result; }

	  private:
		Tensor m_result;
	};

} // namespace Ritsu