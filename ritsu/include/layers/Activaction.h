/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2025 Valdemar Lindberg
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
#include "Layer.h"

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class Activaction : public Layer<float> {
	  public:
		Activaction(const std::string &name = "activaction") : Layer(name) {}
		~Activaction() override = default;

		// virtual Tensor<float> operator<<(Tensor<float> &tensor) override { return tensor; }
		//
		// virtual Tensor<float> operator>>(Tensor<float> &tensor) override { return tensor; }
		//
		// virtual Tensor<float> &operator()(Tensor<float> &tensor) override { return tensor; }
	};
} // namespace Ritsu