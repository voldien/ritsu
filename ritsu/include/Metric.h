#pragma once
#include "Math.h"
#include "Tensor.h"
#include <string>

namespace Ritsu {

	class Metric {
	  public:
		Metric(const std::string &m_name) { this->name = m_name; }

		virtual void update_state(const Tensor &tensor) = 0;

		virtual void reset_state() = 0;

		virtual const Tensor &result() const = 0;

		const std::string getName() const { return this->name; }

	  private:
		std::string name;
	};

	class MetricAccuracy : public Metric {
	  public:
		MetricAccuracy(const std::string &name = "accuracy") : Metric(name) { this->reset_state(); }

		void update_state(const Tensor &tensor) override {}

		void reset_state() override { m_result = Tensor({1}); }

		const Tensor &result() const override { return this->m_result; }

	  private:
		Tensor m_result;
	};

	class MetricMean : public Metric {
	  public:
		MetricMean(const std::string &name = "mean") : Metric(name) { this->reset_state(); }

		void update_state(const Tensor &tensor) override {
			m_result.getValue<float>(0) = Math::mean(tensor.getRawData<float>(), tensor.getNrElements());
		}

		void reset_state() override { m_result = Tensor({1}); }

		const Tensor &result() const override { return this->m_result; }

	  private:
		Tensor m_result;
	};

} // namespace Ritsu