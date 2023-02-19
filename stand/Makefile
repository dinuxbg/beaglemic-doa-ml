
all: beaglemic-stand.stl beaglemic-stand.png

%.stl: %.scad
	openscad -o $@ --render stl $^

%.png: %.scad
	openscad -o $@ $^

.PHONY: all
