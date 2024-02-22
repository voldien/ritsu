/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2023 Valdemar Lindberg
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 */
#pragma once
#include "Object.h"
#include "Tensor.h"
#include <cmath>
#include <string>

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class Metric : public Object {
	  public:
		Metric(const std::string &m_name) : Object(m_name) {}

		virtual void update_state(const std::initializer_list<Tensor<float>> args) = 0;

		template <typename... Args> void update_state(Args... args) {
			return this->update_state({std::forward<Args>(args)...});
		}

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

		void update_state(const std::initializer_list<Tensor<float>> args) override { /*	*/

			assert(args.size() >= 2);

			/*	*/
			const Tensor<float> *refA = &(*args.begin());
			const Tensor<float> *refB = &(*(args.begin() + 1));

			size_t correct = 0;
			for (size_t i = 0; i < refA->getNrElements(); i++) {
				
				/*	*/
				if (refA->getValue<float>(i) == refB->getValue<float>(i)) {
					correct++;
				}
			}

			this->m_result.getValue<float>(0) = static_cast<float>(correct) / static_cast<float>(refA->getNrElements());
		}

		void reset_state() override { this->m_result = Tensor<float>({1}); }

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

		void update_state(const std::initializer_list<Tensor<float>> args) override {

			assert(args.size() >= 1);

			const Tensor<float> *refA = &(*args.begin());

			m_result.getValue<float>(0) = refA->mean();
			assert(!std::isnan(m_result.getValue<float>(0)));
		}

		void reset_state() override {
			this->m_result = Tensor<float>({1});
			this->m_result.getValue<float>(0) = 10000000;
		}

		const Tensor<float> &result() const noexcept override { return this->m_result; }

	  private:
		Tensor<float> m_result;
	};

} // namespace Ritsu