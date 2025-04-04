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

    mkl_free(J);
    mkl_free(JR);
    mkl_free(J1);
    mkl_free(J2);
    mkl_free(Res);
    mkl_free(ut);
  }
  mkl_free(t);
}

int CForwardEuler(double TL, double TR, double *u0, int s, int n, double *uPred,
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

  tbb::parallel_for(0, N, [&uPred, &N, &M, &TT](int i) {
    double uu0[3];
    for (int ii = 0; ii < 3; ii++) {
      uu0[ii] = uPred[ii * (N + 1) + i];
    }
    double u1[3];
    double V[9];
    CForwardEuler(TT[i], TT[i + 1], uu0, 3, M, uPred, u1, V);
    std::cout << std::this_thread::get_id() << " ";
    std::cout << "u1: " << u1[0] << " " << u1[1] << " " << u1[2] << " ";
    std::cout << std::endl;
  });

  mkl_free(Unew);

  mkl_free(TT);
}

int main() {
  int M = 10;
  double T = 1;
  double u0[3] = {20, 5, -5};
  int K = 9;
  int N = 500;
  double *uPred = new double[(N + 1) * 3];
  ForwardEuler(0, T, u0, 3, N, uPred);

  double *RES = (double *)mkl_malloc((N + 1) * 3 * sizeof(double), 64);

  MultipleShooting(0, T, u0, N, K, M, uPred, RES);

  mkl_free(RES);

  return 0;
}