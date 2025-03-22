// Created by Kai Huang on 2025/03/16
// Email: huangkai23@mails.jlu.edu.cn
// Description: ForwardEuler in parallel with PETSc.
// P49
#include "mpi.h"
#include "petscerror.h"
#include "petsclog.h"
#include "petscmath.h"
#include "petscsys.h"
#include "petscsystypes.h"
#include "petscvec.h"
#include <cmath>
#include <cstddef>
#include <iostream>
#include <map>
#include <petsc.h>

PetscMPIInt rank;
PetscMPIInt size;

double f(double t, double u) { return PetscCosReal(t) * u; }

int ForwardEuler(PetscReal TL, PetscReal TR, PetscReal u0, PetscInt N,
                 PetscReal *u) {
  PetscReal dt = (TR - TL) / N;
  PetscReal *t = new PetscReal[N + 1];
  double tt = TL;
  for (int i = 0; i <= N; i++) {
    t[i] = tt;
    tt += dt;
  }

  u[0] = u0;

  for (int n = 1; n <= N; n++) {
    double tvalues = t[n - 1];
    double unew = u0 + dt * f(tvalues, u0);
    u[n] = unew;
    u0 = unew;
  }
  delete[] t;
  return 0;
}

int FindClosesInterval(double x, const Vec &y) {}

void Nievergelt(PetscInt nSteps, const PetscReal T, PetscReal u0,
                const PetscInt N, PetscReal *uPred, const PetscInt Mn,
                const PetscReal width, Vec &U,
                std::map<int, std::map<int, Vec>> &uTraj) {
  PetscReal *TT = new PetscReal[N + 1];
  PetscReal dt = T / N;
  if (rank == 0) {
    double TL = 0;
    double TR = 0 + dt;
    PetscReal *u = new PetscReal[nSteps + 1];
    ForwardEuler(TL, TR, u0, nSteps, u);

    delete[] u;
  }
  delete[] TT;
}

int main(int argc, char **argv) {

  PetscCall(PetscInitialize(&argc, &argv, NULL,
                            "Compute ForwardEuler with PETSc.\n\n"));
  PetscCall(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  PetscInt N = 10;
  PetscReal TL = 0;
  PetscReal TR = 2 * PETSC_PI;
  PetscReal u0 = 1;
  PetscInt nSteps = 100;
  PetscInt Mn = 2;
  PetscReal width = 0.75;

  PetscReal *uPred = new PetscReal[N + 1];
  if (rank == 0) {
    ForwardEuler(TL, TR, u0, N, uPred);
  }

  PetscCall(MPI_Bcast(uPred, N + 1, MPIU_REAL, 0, PETSC_COMM_WORLD));

  Vec U;
  std::map<int, std::map<int, Vec>> uTraj;

  Nievergelt(nSteps, TR, u0, N, uPred, Mn, width, U, uTraj);

  PetscCall(PetscFinalize());
  VecDestroy(&U);
  for (auto &u : uTraj) {
    for (auto &v : u.second) {
      VecDestroy(&v.second);
    }
  }
  PetscFree(uPred);
  return 0;
}
