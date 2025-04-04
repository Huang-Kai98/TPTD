// Created by Kai Huang on 2025/03/23
// Email: huangkai23@mails.jlu.edu.cn
// Description: ForwardEuler in parallel with PETSc.
// P49
// Lorenz equation
// dx/dt = sigma(y-x)  x(0)=x_0
// dy/dt = x(r-z)-y    y(0)=y_0
// dz/dt = xy-bz       z(0)=z_0
// sigma = 10, r = 28, b = 8/3
// x0 = 20, y0 = 5, z0 = -5

#include "mpi.h"
#include "petscerror.h"
#include "petsclog.h"
#include "petscmat.h"
#include "petscmath.h"
#include "petscsftypes.h"
#include "petscsys.h"
#include "petscsystypes.h"
#include "petscvec.h"
#include <cmath>
#include <cstddef>
#include <iostream>
#include <map>
#include <petsc.h>

void f(PetscReal t, PetscReal *u, PetscReal *Res) {
  PetscReal sigma = 10;
  PetscReal r = 28;
  PetscReal b = 8.0 / 3.0;
  Res[0] = sigma * (u[1] - u[0]);
  Res[1] = r * u[0] - u[1] - u[0] * u[2];
  Res[2] = u[0] * u[1] - b * u[2];
}

void jac(PetscReal t, PetscReal *u, PetscReal *J) {
  PetscReal sigma = 10;
  PetscReal r = 28;
  PetscReal b = 8.0 / 3.0;
  J[0] = -sigma;
  J[1] = r - u[2];
  J[2] = u[1];
  J[3] = sigma;
  J[4] = -1;
  J[5] = u[0];
  J[6] = 0;
  J[7] = -u[0];
  J[8] = -b;
}

void ForwardEuler(PetscReal TL, PetscReal TR, PetscReal *u0, PetscInt n,
                  PetscInt N, PetscReal *u) {
  PetscReal dt = (TR - TL) / N;
  PetscReal *t = new PetscReal[N + 1];
  double tt = TL;
  for (int i = 0; i <= N; i++) {
    t[i] = tt;
    tt += dt;
  }

  for (int j = 0; j < n; j++) {
    u[j * (N + 1) + 0] = u0[j];
  }

  for (int i = 1; i <= N; i++) {
    PetscReal *Res = new PetscReal[n];
    PetscReal *ut = new PetscReal[n];
    for (int j = 0; j < n; j++) {
      ut[j] = u[j * (N + 1) + i - 1];
    }
    f(t[i - 1], ut, Res);
    for (int j = 0; j < n; j++) {
      u[j * (N + 1) + i] = u[j * (N + 1) + i - 1] + dt * Res[j];
    }
    delete[] Res;
    delete[] ut;
  }
}

void ForwardEulerJac(PetscReal TL, PetscReal TR, PetscReal *u0, PetscInt n,
                     PetscInt N, PetscReal *u) {
  PetscReal dt = (TR - TL) / N;
  PetscReal *t = new PetscReal[N + 1];
  double tt = TL;
  for (int i = 0; i <= N; i++) {
    t[i] = tt;
    tt += dt;
  }

  for (int j = 0; j < n * (n + 1); j++) {
    u[j * (N + 1) + 0] = u0[j];
  }
  for (int i = 1; i <= N; i++) {
    PetscReal *Res = new PetscReal[n];
    PetscReal *ut = new PetscReal[n];
    PetscReal *J1 = new PetscReal[n * n];
    PetscReal *J2 = new PetscReal[n * n];
    PetscReal *J = new PetscReal[n * n];
    PetscReal *JR = new PetscReal[n * (n + 1)];
    for (int j = 0; j < n * n; j++) {
      J[j] = 0;
    }

    for (int j = 0; j < n; j++) {
      ut[j] = u[j * (N + 1) + i - 1];
    }
    f(t[i - 1], ut, Res);
    jac(t[i - 1], ut, J1);
    for (int j = n; j < n * (n + 1); j++) {
      J2[j - n] = u[j * (N + 1) + i - 1];
      // std::cout << u[j * (N + 1) + i - 1] << std::endl;
    }

    for (int ii = 0; ii < n; ii++) {
      for (int jj = 0; jj < n; jj++) {
        for (int kk = 0; kk < n; kk++) {
          J[ii + jj * n] += J1[ii + kk * n] * J2[kk + jj * n];
          // std::cout << J2[ii + jj * n] << std::endl;
          //     J[ii * n + jj] += J1[ii * n + kk] * J2[kk * n + jj];
        }
      }
    }

    for (int j = 0; j < n; j++) {
      JR[j] = Res[j];
      // std::cout << JR[j] << std::endl;
    }
    for (int j = n; j < n * (n + 1); j++) {
      JR[j] = J[j - n];
    }

    for (int j = 0; j < n * (n + 1); j++) {
      u[j * (N + 1) + i] = u[j * (N + 1) + i - 1] + dt * JR[j];
    }

    delete[] J;
    delete[] J2;
    delete[] J1;
    delete[] Res;
    delete[] ut;
  }
}

int CForwardEuler(PetscReal TL, PetscReal TR, PetscReal *u0, int s, PetscInt n,
                  PetscReal *uPred, PetscReal *u1, PetscReal *V) {

  double *u0All = new PetscReal[s * (s + 1)];
  for (int i = 0; i < s; i++) {
    u0All[i] = u0[i];
  }
  for (int i = s; i < s * (s + 1); i++) {
    u0All[i] = 0;
  }
  for (int i = 0; i < s; i++) {
    u0All[s + i * s + i] = 1;
  }

  double *u = new PetscReal[s * (s + 1) * (n + 1)];

  ForwardEulerJac(TL, TR, u0All, 3, n, u);

  for (int i = 0; i < s; i++) {
    u1[i] = u[i * (n + 1) + n];
  }
  for (int i = s; i < s * (s + 1); i++) {
    V[i - s] = u[i * (n + 1) + n];
  }

  /*
  for (int i = 0; i < 11; i++) {
    for (int j = 0; j < 12; j++) {
      std::cout << u[j * 11 + i] << " ";
    }
    std::cout << std::endl;
  }
  */

  delete[] u;
  delete[] u0All;
  return 0;
};

int MultipleShooting(PetscReal TL, PetscReal TR, PetscReal *u0, PetscInt N,
                     PetscInt K, PetscInt M, PetscReal *uPred, Mat &U) {
  PetscMPIInt rank;
  PetscMPIInt size;
  PetscCall(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscReal dt = (TR - TL) / N;
  PetscReal *TT = new PetscReal[N + 1];
  for (int i = 0; i <= N; i++) {
    TT[i] = TL + i * dt;
  }

  PetscInt rstart, rend;
  MatGetOwnershipRange(U, &rstart, &rend); // 获取 mat1 的行范围
  for (int i = rstart; i < rend; i++) {
    MatSetValue(U, i, 0, uPred[i], INSERT_VALUES);
  }
  if (rank == 0) {
    PetscInt *col = new PetscInt[3];
    for (int k = 0; k < K; k++) {
      for (int i = 0; i < 3; i++) {
        col[i] = i * (N + 1);
      }
      MatSetValues(U, 3, col, 1, &k, u0, INSERT_VALUES);
    }

    delete[] col;
  }
  MatAssemblyBegin(U, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(U, MAT_FINAL_ASSEMBLY);

  PetscReal *u1 = new PetscReal[3];
  PetscReal *V = new PetscReal[9];

  Mat Mu;
  MatCreate(PETSC_COMM_WORLD, &Mu);
  MatSetSizes(Mu, PETSC_DECIDE, PETSC_DECIDE, N, 3);
  MatSetFromOptions(Mu);
  MatSetUp(Mu);
  Mat MV;
  MatCreate(PETSC_COMM_WORLD, &MV);
  MatSetSizes(MV, PETSC_DECIDE, PETSC_DECIDE, N, 9);
  MatSetFromOptions(MV);
  MatSetUp(MV);

  PetscInt u1col[3] = {0, 1, 2};
  PetscInt Vcol[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};

  MatGetOwnershipRange(Mu, &rstart, &rend); // 获取 mat1 的行范围

  for (int i = rstart; i < rend; i++) {
    double *uu0 = new PetscReal[3];
    for (int ii = 0; ii < 3; ii++) {
      uu0[ii] = uPred[(N + 1) * ii + i];
    }

    CForwardEuler(TT[i], TT[i + 1], uu0, 3, M, uPred, u1, V);

    MatSetValues(Mu, 1, &i, 3, u1col, u1, INSERT_VALUES);
    MatSetValues(MV, 1, &i, 9, Vcol, V, INSERT_VALUES);
    delete[] uu0;
  }

  MatAssemblyBegin(Mu, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Mu, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(MV, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(MV, MAT_FINAL_ASSEMBLY);

  Vec UK;
  VecCreate(PETSC_COMM_WORLD, &UK);
  VecSetSizes(UK, PETSC_DECIDE, (N + 1) * 3);
  VecSetFromOptions(UK);
  VecSetUp(UK);
  VecGetOwnershipRange(UK, &rstart, &rend); // 获取 mat1 的行范围

  for (int i = rstart; i < rend; i++) {
    VecSetValue(UK, i, uPred[i], INSERT_VALUES);
  }

  VecAssemblyBegin(UK);
  VecAssemblyEnd(UK);
  MatGetOwnershipRange(Mu, &rstart, &rend);
  PetscReal *UK1 = new PetscReal[3];
  K = 2;
  for (int k = 0; k < K; k++) {
    Vec UK_seq;
    VecScatter ctx;
    VecScatterCreateToAll(UK, &ctx, &UK_seq);
    VecScatterBegin(ctx, UK, UK_seq, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(ctx, UK, UK_seq, INSERT_VALUES, SCATTER_FORWARD);
    for (int i = 0; i < 3; i++) {
      UK1[i] = u0[i];
    }
    for (int n = rstart; n < rend; n++) {
      PetscReal *Mu1 = new PetscReal[3];
      PetscReal *MV1 = new PetscReal[9];
      MatGetValues(Mu, 1, &n, 3, u1col, Mu1);
      MatGetValues(MV, 1, &n, 9, Vcol, MV1);

      PetscReal *UKn = new PetscReal[3];
      PetscInt *UKcol = new PetscInt[3];
      for (int i = 0; i < 3; i++) {
        UKcol[i] = i * (N + 1) + n;
      }
      VecGetValues(UK_seq, 3, UKcol, UKn);

      for (int i = 0; i < 3; i++) {
        UK1[i] = UK1[i] - UKn[i];
      }

      PetscReal *TT = new PetscReal[3];
      for (int i = 0; i < 3; i++) {
        TT[i] = 0;
      }
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          TT[i] += MV1[j * 3 + i] * UK1[j];
        }
      }

      for (int i = 0; i < 3; i++) {
        UK1[i] = Mu1[i] + TT[i];
      }
      if (k == 1 && n == 0) {
        for (int i = 0; i < 3; i++) {
          std::cout << UK1[i] << "  ";
        }
        std::cout << std::endl;
      }
      for (int i = 0; i < 3; i++) {
        UKcol[i] = i * (N + 1) + n + 1;
      }
      PetscInt UKINDE = k + 1;
      MatSetValues(U, 3, UKcol, 1, &UKINDE, UK1, INSERT_VALUES);
      VecSetValues(UK, 3, UKcol, UK1, INSERT_VALUES);

      delete[] TT;
      delete[] Mu1;
      delete[] MV1;
      delete[] UKcol;
      delete[] UKn;
    }
    VecDestroy(&UK_seq);
    VecScatterDestroy(&ctx);
  }
  delete[] UK1;
  MatAssemblyBegin(U, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(U, MAT_FINAL_ASSEMBLY);
  // MatView(U, PETSC_VIEWER_STDOUT_WORLD);
  /*

  MatGetOwnershipRange(Mu, &rstart, &rend);
  PetscReal *UK1 = new PetscReal[(N + 1) * 3];
  for (int k = 0; k < K; k++) {
    Vec UK_seq;
    VecScatter ctx;
    VecScatterCreateToAll(UK, &ctx, &UK_seq);
    VecScatterBegin(ctx, UK, UK_seq, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(ctx, UK, UK_seq, INSERT_VALUES, SCATTER_FORWARD);
    for (int i = 0; i < 3; i++) {
      UK1[i * (N + 1)] = u0[i];
    }
    for (int n = rstart; n < rend; n++) {
      MatGetValues(Mu, 1, &n, 3, u1col, u1);
      MatGetValues(MV, 1, &n, 9, Vcol, V);
      PetscReal *UK1n = new PetscReal[3];
      PetscInt *UKcol = new PetscInt[3];
      for (int i = 0; i < 3; i++) {
        UKcol[i] = i * (N + 1) + n;
      }
      VecGetValues(UK_seq, 3, UKcol, UK1n);
      for (int i = 0; i < 3; i++) {
        std::cout << UK1n[i] << "  ";
      }
      std::cout << std::endl;

      free(UKcol);
      free(UK1n);
    }
    VecDestroy(&UK_seq);
    VecScatterDestroy(&ctx);
  }
  free(UK1);
  */
  VecDestroy(&UK);

  MatDestroy(&Mu);
  MatDestroy(&MV);

  free(u1);
  free(V);

  free(TT);
  return 0;
};

int main(int argc, char **argv) {

  PetscCall(PetscInitialize(&argc, &argv, NULL,
                            "Compute ForwardEuler with PETSc.\n\n"));

  PetscInt M = 10;
  PetscReal T = 1;
  PetscReal u0[3] = {20, 5, -5};
  PetscInt K = 9;
  PetscInt N = 500;

  PetscReal *uPred = new PetscReal[(N + 1) * 3];

  ForwardEuler(0, T, u0, 3, N, uPred);

  // PetscReal *U = new PetscReal[(N + 1) * 3 * (K + 1)];
  Mat U;
  MatCreate(PETSC_COMM_WORLD, &U);
  MatSetSizes(U, PETSC_DECIDE, PETSC_DECIDE, (N + 1) * 3, K + 1);
  MatSetFromOptions(U);
  MatSetUp(U);

  MultipleShooting(0, T, u0, N, K, M, uPred, U);
  MatDestroy(&U);
  PetscCall(PetscFinalize());
  free(uPred);
  // free(U);
  return 0;
}
