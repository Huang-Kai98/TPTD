/*
* ----------------------------------------------------------------------------
 * Project Name : MultipleShooting
 * File         : MultipleShooting.cpp
 * Author       : Nero <huangkai23@mails.jlu.edu.cn>
 * Created      : 2025-04-04
 * Description  :  N equidistant coarse time points doing K iterations. It uses a
 * propagator P(t0,t1,ut0) that returns the solution and the Jacobian
 * at time t1, and an initial guess uPred containing N+1 starting
 * values for the algorithm.
 *
 * Copyright (c) 2025 Nero
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 * ----------------------------------------------------------------------------
 */


#include <iostream>
#include <mkl.h>
#include <mkl_service.h>
#include <oneapi/tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

void f(double t, double *u, double *Res) {
  double sigma = 10;
  double r = 28;
  double b = 8.0 / 3.0;
  Res[0] = sigma * (u[1] - u[0]);
  Res[1] = r * u[0] - u[1] - u[0] * u[2];
  Res[2] = u[0] * u[1] - b * u[2];
}

void jac(double t, double *u, double *J) {
  double sigma = 10;
  double r = 28;
  double b = 8.0 / 3.0;
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

void ForwardEuler(double TL, double TR, double *u0, int n, int N, double *u) {
  double dt = (TR - TL) / N;
  double *t = (double *)mkl_malloc((N + 1) * sizeof(double), 64);
  double tt = TL;
  for (int i = 0; i <= N; i++) {
    t[i] = tt;
    tt += dt;
  }

  for (int j = 0; j < n; j++) {
    u[j * (N + 1) + 0] = u0[j];
  }

  for (int i = 1; i <= N; i++) {
    double *Res = (double *)mkl_malloc(n * sizeof(double), 64);
    double *ut = (double *)mkl_malloc(n * sizeof(double), 64);
    for (int j = 0; j < n; j++) {
      ut[j] = u[j * (N + 1) + i - 1];
    }
    f(t[i - 1], ut, Res);
    for (int j = 0; j < n; j++) {
      u[j * (N + 1) + i] = u[j * (N + 1) + i - 1] + dt * Res[j];
    }
    mkl_free(Res);
    mkl_free(ut);
  }
  mkl_free(t);
}

void ForwardEulerJac(double TL, double TR, double *u0, int n, int N,
                     double *u) {
  double dt = (TR - TL) / N;

  double *t = (double *)mkl_malloc((N + 1) * sizeof(double), 64);
  double tt = TL;
  for (int i = 0; i <= N; i++) {
    t[i] = tt;
    tt += dt;
  }

  for (int j = 0; j < n * (n + 1); j++) {
    u[j * (N + 1) + 0] = u0[j];
  }
  for (int i = 1; i <= N; i++) {
    double *Res = (double *)mkl_malloc(n * sizeof(double), 64);
    double *ut = (double *)mkl_malloc(n * sizeof(double), 64);
    double *J1 = (double *)mkl_malloc(n * n * sizeof(double), 64);
    double *J2 = (double *)mkl_malloc(n * n * sizeof(double), 64);
    double *J = (double *)mkl_calloc(n * n, sizeof(double), 64);
    double *JR = (double *)mkl_malloc(n * (n + 1) * sizeof(double), 64);

    for (int j = 0; j < n; j++) {
      ut[j] = u[j * (N + 1) + i - 1];
    }
    f(t[i - 1], ut, Res);
    jac(t[i - 1], ut, J1);
    for (int j = n; j < n * (n + 1); j++) {
      J2[j - n] = u[j * (N + 1) + i - 1];
    }

    for (int ii = 0; ii < n; ii++) {
      for (int jj = 0; jj < n; jj++) {
        for (int kk = 0; kk < n; kk++) {
          J[ii + jj * n] += J1[ii + kk * n] * J2[kk + jj * n];
        }
      }
    }

    for (int j = 0; j < n; j++) {
      JR[j] = Res[j];
    }
    for (int j = n; j < n * (n + 1); j++) {
      JR[j] = J[j - n];
    }

    for (int j = 0; j < n * (n + 1); j++) {
      u[j * (N + 1) + i] = u[j * (N + 1) + i - 1] + dt * JR[j];
    }

    mkl_free(J);
    mkl_free(JR);
    mkl_free(J1);
    mkl_free(J2);
    mkl_free(Res);
    mkl_free(ut);
  }
  mkl_free(t);
}

int CForwardEuler(double TL, double TR, double *u0, int s, int n,
                  double *u1, double *V) {

  double *u0All = (double *)mkl_calloc(s * (s + 1), sizeof(double), 64);

  for (int i = 0; i < s; i++) {
    u0All[i] = u0[i];
  }
  for (int i = 0; i < s; i++) {
    u0All[s + i * s + i] = 1;
  }

  double *u = (double *)mkl_malloc(s * (s + 1) * (n + 1) * sizeof(double), 64);

  ForwardEulerJac(TL, TR, u0All, 3, n, u);

  for (int i = 0; i < s; i++) {
    u1[i] = u[i * (n + 1) + n];
  }
  for (int i = s; i < s * (s + 1); i++) {
    V[i - s] = u[i * (n + 1) + n];
  }

  mkl_free(u);
  mkl_free(u0All);
  return 0;
};

void MultipleShooting(double TL, double TR, double *u0, int N, int K, int M,
                      double *uPred, double *U) {
  double dt = (TR - TL) / N;
  double *TT = (double *)mkl_malloc((N + 1) * sizeof(double), 64);
  for (int i = 0; i <= N; i++) {
    TT[i] = TL + i * dt;
  }
  for (int i = 0; i < 3; i++) {
    U[i * (N + 1)] = u0[i];
  }
  cblas_dcopy(3 * (N + 1), uPred, 1, U, 1);
  double *Unew = (double *)mkl_malloc(3 * (N + 1) * sizeof(double), 64);
  double *u=(double *)mkl_malloc(3 * N  * sizeof(double), 64);
  double *V=(double *)mkl_malloc(9 * N * sizeof(double), 64);

  for (int k=0;k<K;k++) {

    tbb::parallel_for(0, N, [&U,&u,&V, &N, &M, &TT](int i) {
      double uu0[3];
      for (int ii = 0; ii < 3; ii++) {
        uu0[ii] = U[ii * (N + 1) + i];
      }
      double u1[3];
      double V1[9];
      CForwardEuler(TT[i], TT[i + 1], uu0, 3, M, u1, V1);
      cblas_dcopy(3,u1,1,u+3*i,1);
      cblas_dcopy(9,V1,1,V+9*i,1);
    });

    for (int ii = 0; ii < 3; ii++) {
      Unew[ii * (N + 1)] = u0[ii];
    }
    for (int n=0;n<N;n++) {
      double temp[3],temp1[3];
      for (int ii = 0; ii < 3; ii++) {
        temp[ii] = Unew[ii * (N + 1) + n]-U[ii * (N + 1) + n];
      }

      cblas_dgemv(CblasColMajor,CblasNoTrans,3,3,1.0,V+9*n,3,temp,1,0.0,temp1,1);

      cblas_daxpy(3,1.0,u+3*n,1,temp1,1);
      for (int ii = 0; ii < 3; ii++) {
        Unew[ii * (N + 1)+n+1] = temp1[ii];
      }
    }
    cblas_dcopy((N+1)*3,Unew,1,U,1);
  }

  for (int i=0;i<N+1;i++) {
    std::cout<<U[0 * (N + 1)+i]<<" "<< U[1 * (N + 1)+i]<<" "<< U[2 * (N + 1)+i]<<std::endl;
  }

  mkl_free(V);
  mkl_free(u);
  mkl_free(Unew);

  mkl_free(TT);
}

int main() {
  int M = 10;
  double T = 1;
  double u0[3] = {20, 5, -5};
  int K = 9;
  int N = 500;
  double *uPred = (double *)mkl_malloc((N + 1) * 3 * sizeof(double), 64);
  ForwardEuler(0, T, u0, 3, N, uPred);

  double *RES = (double *)mkl_malloc((N + 1) * 3 * sizeof(double), 64);

  MultipleShooting(0, T, u0, N, K, M, uPred, RES);

  mkl_free(RES);
  mkl_free(uPred);

  return 0;
}