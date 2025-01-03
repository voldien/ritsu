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
#include <string>

namespace Ritsu {

	/**
	 * @brief
	 *
	 */
	class Object {
	  public:
		Object(const std::string &newName) noexcept : name(newName) {}

		const std::string &getName() const noexcept { return this->name; }
		void setName(const std::string &newName) noexcept { this->name = newName; }

	  private:
		std::string name;
	};

} // namespace Ritsu