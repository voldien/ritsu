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
	 * 
	 * @tparam T 
	 */
	template <typename T> class Adam : public Optimizer<T> {
	  public:
		Adam(const T learningRate, const T beta_1, const T beta_2, const std::string &name = "adam")
			: Optimizer<T>(learningRate, name) {
			this->beta_1 = beta_1;
			this->beta_2 = beta_2;
		}

	  private:
		T beta_1;
		T beta_2;
	};

} // namespace Ritsu