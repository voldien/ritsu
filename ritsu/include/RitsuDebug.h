
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
#include "Ritsu.h"
#include <cstdlib>

namespace Ritsu {

	static bool isDebugEnabled() {
		static const char *debug_value_env = std::getenv("RITSU_DEBUG");
		if (debug_value_env == nullptr) {
			return false;
		}
		return std::strcmp(std::getenv("RITSU_DEBUG"), "1") == 0;
	}

	static void enableDebug() { setenv("RITSU_DEBUG", "1", 0); }

	/*	*/
	template <typename U = float> std::ostream &debug_print_layer(std::ostream &stream, Layer<U> &layer) noexcept {

		if (isDebugEnabled()) {
			stream << "Layer: " << layer.getName() << std::endl;

			if (layer.getTrainableWeights().has_value() && !layer.getTrainableWeights().value().empty()) {
				// stream << std::endl << "trainable: " << layer.getTrainableWeights().value() << std::endl <<
				// std::endl;
			}

			if (layer.getVariables().has_value() && !layer.getVariables().value().empty()) {
				// stream << std::endl << "non-trainable: " << layer.getVariables().value() << std::endl << std::endl;
			}
		}

		return stream;
	}

	/*	*/
	template <typename U = std::float_t>
	std::ostream &debug_print_tensor(std::ostream &stream, Tensor<U> &tensor, const std::string &name = "") noexcept {
		if (isDebugEnabled()) {
			stream << std::endl << name << tensor << std::endl << std::endl;
		}
		return stream;
	}

	/*	*/
	template <typename U = std::float_t>
	std::ostream &debug_print_tensor_layer(std::ostream &stream, const Layer<U> &layer, Tensor<U> &tensor) noexcept {
		if (isDebugEnabled()) {
			stream << std::endl << layer.getName() << " " << tensor.getShape() << std::endl << tensor << std::endl << std::endl;
		}
		return stream;
	}

} // namespace Ritsu
