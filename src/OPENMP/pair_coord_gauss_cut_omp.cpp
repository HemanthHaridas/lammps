/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   This software is distributed under the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Hemanth Haridas (U. of U.), Maxime Pouvreau (PNNL)
------------------------------------------------------------------------- */

#include "pair_coord_gauss_cut_omp.h"

#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neigh_list.h"
#include "suffix.h"
#include "math_const.h"

#include <iostream>
#include <cmath>

#include "omp_compat.h"
using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairCoordGaussCutOMP::PairCoordGaussCutOMP(LAMMPS *lmp) :
  PairCoordGaussCut(lmp), ThrOMP(lmp, THR_PAIR)
{
  suffix_flag |= Suffix::OMP;
  respa_enable = 0;
  comm_forward = 1;
  comm_reverse = 1;
}

/* ---------------------------------------------------------------------- */

void PairCoordGaussCutOMP::compute(int eflag, int vflag)
{
  ev_init(eflag,vflag);

  const int nall = atom->nlocal + atom->nghost;
  const int nthreads = comm->nthreads;
  const int inum = list->inum;

#if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE LMP_SHARED(eflag,vflag)
#endif
  {
    int ifrom, ito, tid;

    loop_setup_thr(ifrom, ito, tid, inum, nthreads);
    ThrData *thr = fix->get_thr(tid);
    thr->timer(Timer::START);
    ev_setup_thr(eflag, vflag, nall, eatom, vatom, nullptr, thr);

    if (evflag) {
      if (eflag) {
        if (force->newton_pair) eval<1,1,1>(ifrom, ito, thr);
        else eval<1,1,0>(ifrom, ito, thr);
      } else {
        if (force->newton_pair) eval<1,0,1>(ifrom, ito, thr);
        else eval<1,0,0>(ifrom, ito, thr);
      }
    } else {
      if (force->newton_pair) eval<0,0,1>(ifrom, ito, thr);
      else eval<0,0,0>(ifrom, ito, thr);
    }

    thr->timer(Timer::PAIR);
    reduce_thr(this, eflag, vflag, thr);
  } // end of omp parallel region
}

template <int EVFLAG, int EFLAG, int NEWTON_PAIR>
void PairCoordGaussCutOMP::eval(int iifrom, int iito, ThrData * const thr)
{
  // all this is copied from the gauss/cut/omp version   
  int i,j,ii,jj,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq,r,rexp,ugauss,factor_lj;
  int *ilist,*jlist,*numneigh,**firstneigh;

  // These are new variables required for coord/gauss/cut
  double factor_coord, coord_nr, coord_dr;

  // this is required for the coordination number calculation
  int n_ii = iito - iifrom + 1;
  std::cout << n_ii << "\n";


  evdwl = 0.0;

  const auto * _noalias const x = (dbl3_t *) atom->x[0];
  auto * _noalias const f = (dbl3_t *) thr->get_f()[0];
  const int * _noalias const type = atom->type;
  const int nlocal = atom->nlocal;
  const double * _noalias const special_lj = force->special_lj;
  double fxtmp,fytmp,fztmp;

  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // create a 2D array to hold the coordination numbers
  double coord_tmp[n_ii][atom->ntypes+1];
  for (ii = 0; ii < n_ii; ii++)
  {
    for (int ij = 0; ij < atom->ntypes+1; ij++)
    {
        coord_tmp[ii][ij] = 0;
    }
  }
  
  // loop over neighbors of my atoms

  for (ii = iifrom; ii < iito; ++ii) {

    i = ilist[ii];
    xtmp = x[i].x;
    ytmp = x[i].y;
    ztmp = x[i].z;
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    fxtmp=fytmp=fztmp=0.0;

    // first loop to get the coordination numbers
    for (jj = 0; jj < jnum; jj++)
    {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j].x;
      dely = ytmp - x[j].y;
      delz = ztmp - x[j].z;
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype] && itype == types[itype][jtype] && jtype == types[jtype][itype] && itype <= jtype)
      {
        r = sqrt(rsq);
        factor_coord = r / rnh[itype][jtype];
        coord_nr = 1 - pow(factor_coord, 6);
        coord_dr = 1 - pow(factor_coord, 12);
        coord_tmp[ii][jtype] = coord_tmp[ii][jtype] + (coord_nr/coord_dr);
      }
    //   std::cout << ii << "\t" << itype << "\t" << jj << "\t" << jtype << "\t" << coord_tmp[ii][jtype] << "\n";
    }
    
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];

      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j].x;
      dely = ytmp - x[j].y;
      delz = ztmp - x[j].z;
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype] && itype == types[itype][jtype] && jtype == types[jtype][itype] && itype <= jtype) {

        // calculate scaling factors
        double pre_exponent_one = coord_tmp[ii][jtype] - coord_low[itype][jtype];
        double pre_exponent_two = coord_tmp[ii][jtype] - coord_high[itype][jtype];
        double scale_factor_one = exp(-1*pre_exponent_one*pre_exponent_one)*weight_low[itype][jtype];
        double scale_factor_two = exp(-1*pre_exponent_two*pre_exponent_two)*weight_high[itype][jtype];
        double scale_factor = (scale_factor_one + scale_factor_two)*hgauss[itype][jtype];

        // now calculate energy
        ugauss = (scale_factor/ sqrt(MY_2PI) / sigmah[i][j]) * exp(-0.5*rexp*rexp);
        fpair = factor_lj*rexp/r*ugauss/sigmah[itype][jtype];

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;
        if (NEWTON_PAIR || j < nlocal) {
          f[j].x -= delx*fpair;
          f[j].y -= dely*fpair;
          f[j].z -= delz*fpair;
        }

        if (EFLAG) {
          evdwl = ugauss - offset[itype][jtype];
          evdwl *= factor_lj;
        }

        if (EVFLAG) ev_tally_thr(this,i,j,nlocal,NEWTON_PAIR,
                                 evdwl,0.0,fpair,delx,dely,delz,thr);
      }
    }
    f[i].x += fxtmp;
    f[i].y += fytmp;
    f[i].z += fztmp;
  }
}

/* ---------------------------------------------------------------------- */

double PairCoordGaussCutOMP::memory_usage()
{
  double bytes = memory_usage_thr();
  bytes += PairCoordGaussCut::memory_usage();

  return bytes;
}
