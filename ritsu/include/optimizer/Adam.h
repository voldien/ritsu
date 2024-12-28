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
#include "Optimizer.h"
#include <map>

namespace Ritsu {

	/**
	 * @brief
	 *
	 * @tparam T
	 */
	template <typename T> class Adam : public Optimizer<T> {
	  public:
		Adam(const T learningRate, const T beta_1 = 0.9, const T beta_2 = 0.999, const T epsilon = 1e-07,
			 const std::string &name = "adam")
			: Optimizer<T>(learningRate, name) {
			this->beta_1 = beta_1;
			this->beta_2 = beta_2;
			this->epsilon = epsilon;
		}
		virtual ~Adam() = default;

		void update_step(const Tensor<T> &gradient, Tensor<T> &variable) override {

			const size_t uid = variable.getUID();
			/*	Init */
			if (m_dw.find(uid) == m_dw.end()) {
				m_dw[uid] = Tensor<T>::zero(variable.getShape());
				v_dw[uid] = Tensor<T>::zero(variable.getShape());
			}

			/*	*/
			m_dw[uid] = (this->m_dw[uid] * this->beta_1) + gradient * (1 - this->beta_1);
			v_dw[uid] = (this->v_dw[uid] * this->beta_2) + (gradient * gradient) * (1 - this->beta_2);

			T t = 1;

			/*	*/
			Tensor<T> m_dw_corr = this->m_dw[uid];
			Tensor<T> v_dw_corr = this->v_dw[uid];

			Tensor<T> gradient_update =
				(m_dw_corr / (v_dw_corr.sqrt() + this->epsilon)) * gradient * -this->getLearningRate();
			this->apply_gradients(gradient_update, variable);
		}

		/**
		 * @brief
		 *
		 * @param gradient
		 * @param variable
		 */
		void apply_gradients(const Tensor<T> &gradient, Tensor<T> &variable) override {
			assert(gradient.getShape() == variable.getShape());

			if (gradient.getShape() == variable.getShape()) {
				variable += gradient;
			} else {
				throw RuntimeException("Invalid Variable and Gradient Shape");
			}
		}

		void build(std::initializer_list<const Tensor<T> &> &list) override { /*	*/ }

	  private:
		T beta_1;
		T beta_2;
		T epsilon;

		std::map<size_t, Tensor<T>> m_dw;
		std::map<size_t, Tensor<T>> v_dw;
	};

} // namespace Ritsu