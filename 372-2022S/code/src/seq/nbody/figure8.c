#include "nbody.h"
#define R 2.0e12
#define sz 20

const double x_min = -R, x_max = R, y_min = -R/2, y_max = R/2;
const int nx = 1000;
const int nstep = 20000;
const int wstep = 10;
const int ncolors = 3;
const int nbodies = 3;
const double delta_t = 1000;

const int colors[][3] = {
  { 255, 0, 0 },
  { 0, 255, 0 },
  { 0, 0, 255 }
};

const Body bodies[] = {
  { 0.97000436e12, -0.24308753e12,  0.466203685e6,  0.43236573e6,  1.499250375e+34, 0, sz },
  { -0.97000436e12,  0.24308753e12,  0.466203685e6,  0.43236573e6,  1.499250375e+34, 1, sz },
  { 0.00000000e12,  0.00000000e12, -0.932407370e6, -0.86473148e6,  1.499250375e+34, 2, sz }
};

/*
3 bodies with equal masses that move around in a figure-8 shape.
This periodic solution was discovered by Cris Moore in 1993.

Creator of the universe: Kevin Wayne

Sample execution: java-introcs NBody 157788000.0 25000.0 < figure8.txt

The data was taken from Table 3.1 of https://info.maths.ed.ac.uk/assets/files/Projects/9.pdf

 0.97000436 -0.24308753  0.466203685  0.43236573
-0.97009436  0.24308753  0.466203685  0.43236573
 0.00000000  0.00000000 -0.932407370 -0.86473146  

and modified in the following ways:
  * Divide masses by G = 6.67e-11 to use the standard gravitational constant (instead of 1).
  * Scale the positions by 10^12; the velocities by 10^6; and the masses by 10^24 in order
    to make the standard values of Tau and dt (157788000.0 and 25000.0) good command-line
    arugments.
*/
