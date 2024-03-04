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
#include <functional>

namespace Ritsu {

	/**
	 * @brief
	 */
	template <typename T> class Ada : public Optimizer<T> {
	  public:
		Ada(const T learningRate, T beta, const std::string &name = "ada") : Optimizer<T>(learningRate, name) {}

		void update_step(const Tensor<T> &gradient, Tensor<T> &variable) {}

		void apply_gradients(const Tensor<T> &gradient, Tensor<T> &variable) {}
	};

} // namespace Ritsu