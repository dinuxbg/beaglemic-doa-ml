// Stand for data acquisition using beaglemic.
// Intended to record audio from different distance
// and angle, which would be used to train DOA AI neural net.
//
// Created 2022 by Dimitar Dimitrov
//
// This work is released with CC0 into the public domain.
// https://creativecommons.org/publicdomain/zero/1.0/

// To align the stand with the source:
//  1. Place a sheet of paper where the stand will be placed. Cardboard is preferred.
//  2. Draw a line between audio source and the sheet of paper.
//  3. Place the stand on the paper, and align the line defined by its center, MIC0 and MIC4 (MIC8). That would set the angle to 0°.
//  4. Fix the center with a metal pin.
//  5. When another angle is required, rotate the stand around the metal pin.
//  6. Always make sure that the drawn "reference" line is aligned with the two opposite corresponding holes of the protractor.


use <./threads.scad>

$fa = 1;
$fs = 0.3;

WALL_W=2;                   // Box wall width.
PCB_SPACER_H = 30;          // Stand to PCB spacer.
PCB_SPACER_R = 3;           // PCB spacer Width Radius.
PCB_MOUNT_HOLE_D = 3;       // Mount hole diameter.
PCB_MOUNT_HOLE_H = PCB_SPACER_H/2;

STAND_OUTER_R = 80;         // Outer ring radius. Ensure it is bigger than the PCB.
STAND_INNER_R = 45;
STAND_BEAM_W = 10;

PROTRACTOR_MARKER_W = 1;    // Protractor marker size.

// difference leaves a small layer if heights are the same.
// So always cut a bit more than the original object.
epsilon = 0.1;

module washer(outer_r, inner_r, w) {
    difference() {
        cylinder(h=w, r=outer_r, center=false);
        translate([0,0,-epsilon]) {
            cylinder(h=w+epsilon*2, r=inner_r, center=false);
        };
    };
};

// Place markers for protractor. The interesting angles are only
// the 5.6° ones (2π/64), which we'll actually train.
module protractor(r, height, length) {
    for(i = [0:64]) {
        rotate([0, 0, i * 360 / 64]) {
            translate([-r, -PROTRACTOR_MARKER_W/2, 0]) {
                l=length - ((i%8)==0 ? 0 : ((i%4)==0 ? 0.4*length : 0.7*length));
                cube([l, PROTRACTOR_MARKER_W, height], center=false);
            };
        };
    };
};

// The inner "washer" is to attach the PCB. The outer is for stability.
difference() {
    washer(STAND_OUTER_R, STAND_OUTER_R - STAND_BEAM_W, WALL_W);
    translate([0,0,-epsilon]) {
        protractor(r=STAND_OUTER_R-1, height=WALL_W + epsilon * 2, length=STAND_BEAM_W/2);
    };
};
washer(STAND_INNER_R, STAND_INNER_R - STAND_BEAM_W, WALL_W);

// Put support beams between the outer and inner washers
for(i = [0:4]) {
    rotate([0, 0, i * 360 / 4]) {
        translate([-STAND_OUTER_R+STAND_BEAM_W/2, -STAND_BEAM_W/2, 0]) {
            cube([STAND_OUTER_R - STAND_INNER_R + STAND_BEAM_W/2, STAND_BEAM_W, WALL_W], false);
        };
    };
};

// Place a mark for MIC0
translate([STAND_INNER_R-STAND_BEAM_W, 0, WALL_W]) {
    linear_extrude(height=1) {
        text(str("->MIC0"), size=6, font="Sans:style=bold", valign="center");
    };
};


// Place PCB holders.
translate([0,0,WALL_W]) {
    for (i = [ [ -32.7, 20.925, 0],
               [ -32.7, -21.075, 0],
               [ 33.3, 20.925, 0],
               [ 33.3, -21.075, 0] ] ) {
        translate(i) {
            ScrewHole(outer_diam=PCB_MOUNT_HOLE_D, height=PCB_MOUNT_HOLE_H, position=[0,0,PCB_SPACER_H - PCB_MOUNT_HOLE_H], rotation=[0,0,0])
                cylinder(h=PCB_SPACER_H, r=PCB_SPACER_R, center=false);
        };
   };
};



// Add pin in the center, to allow pining to a cardboard and rotating in place.
difference() {
    union() {
        translate([0, -STAND_BEAM_W/2, 0]) {
            cube([STAND_INNER_R + STAND_BEAM_W/2, STAND_BEAM_W, WALL_W], false);
        };
        cylinder(h=WALL_W, r=STAND_BEAM_W/2, center=false);
    };
    cylinder(h=WALL_W+1, r=0.5, center=false);
};

// TODO - how to align to the source?
// 1. Laser beam on the stand, callibrated using "iron sights".
// 2. A cord between the center pin and the audio source.
