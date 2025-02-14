/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(coord/gauss/cut,PairCoordGaussCut);
// clang-format on
#else

#ifndef LMP_PAIR_COORD_GAUSS_CUT_H
#define LMP_PAIR_COORD_GAUSS_CUT_H

#include "pair.h"

namespace LAMMPS_NS {

class PairCoordGaussCut : public Pair {
 public:
  PairCoordGaussCut(class LAMMPS *);
  ~PairCoordGaussCut() override;

  void compute(int, int) override;
  void init_style() override;
  
//   double single(int, int, int, int, double, double, double, double &) override;

  void settings(int, char **) override;
  void coeff(int, char **) override;

  double init_one(int, int) override;

  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_restart_settings(FILE *) override;
  void read_restart_settings(FILE *) override;
  void write_data(FILE *fp) override;
  void write_data_all(FILE *fp) override;

  double memory_usage() override;

 protected:
//  Copied from gauss/cut
  double cut_global;
  double **cut;
  double **hgauss, **sigmah, **rmh;
  double **pgauss, **offset;

// Required for coordination number
  double **coord_low; 
  // double **coord_high; 
  // double **weight_low, **weight_high;
  // double **sigma_low, **sigma_high; 
  double **rnh,  **types;
  void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif
