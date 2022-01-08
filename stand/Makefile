
all: beaglemic-stand.stl

%.stl: %.scad
	openscad -o $@ --render stl $^

%.png: %.scad
	openscad -o $@ $^

.PHONY: all
