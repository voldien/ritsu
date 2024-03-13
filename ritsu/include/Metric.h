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
#include "Math.h"
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
		using DType = float;
		Metric(const std::string &m_name) : Object(m_name) {}

		virtual void update_state(const std::initializer_list<const Tensor<DType> *> args) = 0;

		template <typename... Args> void update_state(Args &... args) { return this->update_state({&args...}); }

		virtual void reset_state() = 0;

		virtual const Tensor<DType> &result() const noexcept = 0;

	  protected:
	};

	/**
	 * @brief
	 *
	 */
	class MetricAccuracy : public Metric {
	  public:
		MetricAccuracy(const std::string &name = "accuracy") : Metric(name) { this->reset_state(); }

		void update_state(const std::initializer_list<const Tensor<DType> *> args) override { /*	*/

			assert(args.size() >= 2);

			/*	*/
			const Tensor<DType> *expected_true = (*args.begin());
			const Tensor<DType> *eval_pred = (*(args.begin() + 1));

			assert(expected_true->getShape() == eval_pred->getShape());

			// size_t correct = 0;
			const DType correct = eval_pred->round().equal(*expected_true).sum();

			this->m_result.getValue<DType>(0) = static_cast<DType>(correct) / static_cast<DType>(expected_true->getNrElements());
		}

		void reset_state() override { this->m_result = Tensor<DType>({1}); }

		const Tensor<DType> &result() const noexcept override { return this->m_result; }

	  private:
		Tensor<DType> m_result;
	};

	/**
	 * @brief
	 *
	 */
	class MetricMean : public Metric {
	  public:
		MetricMean(const std::string &name = "mean") : Metric(name) { this->reset_state(); }

		void update_state(const std::initializer_list<const Tensor<DType> *> args) override {

			assert(args.size() >= 1);

			const Tensor<DType> *refA = (*args.begin());

			m_result.getValue<DType>(0) = refA->mean();
			assert(!std::isnan(m_result.getValue<DType>(0)));
		}

		void reset_state() override {
			this->m_result = Tensor<DType>({1});
			this->m_result.getValue<DType>(0) = 10000000;
		}

		const Tensor<DType> &result() const noexcept override { return this->m_result; }

	  private:
		Tensor<DType> m_result;
	};

} // namespace Ritsu