
all: beaglemic-stand.stl

%.stl: %.scad
	openscad -o $@ --render stl $^

.PHONY: all
