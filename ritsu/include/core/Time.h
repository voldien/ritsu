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
#include "../Object.h"
#include <chrono>

namespace Ritsu {

	using namespace std::chrono;

	/**
	 * @brief
	 *
	 */
	class Time : public Object {
	  public:
		Time() : Object("time") {}

		void start() noexcept {
			this->start_timestamp = steady_clock::now();
			this->ticks = steady_clock::now();
		}

		template <typename T> T getElapsed() const noexcept {
			static_assert(std::is_floating_point<T>::value, "Must be a decimal type(float/double/half).");
			duration<T> time_span = duration_cast<duration<T>>(steady_clock::now() - start_timestamp);

			return time_span.count();
		}

		template <typename T> T deltaTime() const noexcept {
			static_assert(std::is_floating_point<T>::value, "Must be a decimal type(float/double/half).");
			duration<T> time_span = duration_cast<duration<T>>(steady_clock::now() - ticks);

			return static_cast<T>(delta_data.count());
		}

		void update() noexcept {

			auto now = steady_clock::now();
			this->delta_data = duration_cast<duration<float>>(now - this->ticks);
			this->ticks = now;
		}

		size_t getTimeResolution() const noexcept {
			return static_cast<size_t>(1.0 / static_cast<double>(std::chrono::high_resolution_clock::period::num) /
									   static_cast<double>(std::chrono::high_resolution_clock::period::den));
		}

	  private: /*  */
		steady_clock::time_point start_timestamp;
		steady_clock::time_point ticks;
		duration<float> delta_data;
	};
} // namespace Ritsu
