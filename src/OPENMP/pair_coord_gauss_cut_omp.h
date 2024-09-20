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

/* ----------------------------------------------------------------------
   Contributing authors: Hemanth Haridas (U. of U.), Maxime Pouvreau (PNNL)
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(coord/gauss/cut/omp,PairCoordGaussCutOMP);
// clang-format on
#else

#ifndef LMP_PAIR_COORD_GAUSS_CUT_OMP_H
#define LMP_PAIR_COORD_GAUSS_CUT_OMP_H

#include "pair_coord_gauss_cut.h"
#include "thr_omp.h"

namespace LAMMPS_NS {

class PairCoordGaussCutOMP : public PairCoordGaussCut, public ThrOMP {

 public:
  PairCoordGaussCutOMP(class LAMMPS *);

  void compute(int, int) override;
  double memory_usage() override;

 private:
  template <int EVFLAG, int EFLAG, int NEWTON_PAIR>
  void eval(int ifrom, int ito, ThrData *const thr);
};

}    // namespace LAMMPS_NS

#endif
#endif
