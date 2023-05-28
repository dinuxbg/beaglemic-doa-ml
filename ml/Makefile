# SPDX-FileCopyrightText: 2022-2023 Dimitar Dimitrov <dimitar@dinux.eu>
# SPDX-License-Identifier: GPL-3.0-or-later

# Build the C++ data parser and augmentation tool.

CXXFLAGS += -O3 -Wall -Wextra
CXXFLAGS += -std=c++20 -mtune=native -march=native
# CXXFLAGS += -fsanitize=address -fsanitize=undefined -fsanitize-address-use-after-scope
# CXXFLAGS += -g3
prepare-data: prepare-data.cc | Makefile
	g++ $(CXXFLAGS) $< -o $@

clean:
	rm -f prepare-data
