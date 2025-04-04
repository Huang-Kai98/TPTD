/*
* ----------------------------------------------------------------------------
 * Project Name : Parareal
 * File         : Parareal.cpp
 * Author       : Nero <Huangkai23@mails.jlu.edu.cn>
 * Created      : 2025-04-04
 * Description  :   PARAREAL implementation of the parareal algorithm
 * U=Parareal(F,G,T,u0,N,K); applies the parareal algorithm with fine
 * solver F(t0,t1,ut0) and coarse solver G(t0,t1,ut0) on [0,T] with
 * initial condition u0 at t=0 using N equidistant coarse time points
 * doing K iterations. The output U{k} contains the parareal
 * approximations at the coarse time points for each iteration k.
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


void ForwardEuler(double TL, double TR,const double *u0, int n, int N, double *u) {
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

void SForwardEuler(double TL,double TR,const double* u0,int n,int N,double* Res) {
  double * u= (double *)mkl_malloc((N+1)* n * sizeof(double), 64);
  ForwardEuler(TL,TR,u0,n,N,u);
  for (int i=0;i<n;i++) {
    Res[i]=u[i*(N+1)+N];
  }
  mkl_free(u);
}

void Parareal(const int F,const int G,const double TL,const double TR,const double *u0,const int N,const int K,double* U) {
  double dt = (TR - TL) / N;
  double *TT = (double *)mkl_malloc((N + 1) * sizeof(double), 64);
  for (int i = 0; i <= N; i++) {
    TT[i] = TL + i * dt;
  }
  cblas_dcopy(3,u0,1,U,1);
  double* Go=(double*)mkl_calloc((N+1)*3,sizeof(double),64);
  for (int n = 1; n <= N; n++) {
    SForwardEuler(TT[n-1],TT[n],U+(n-1)*3,3,G,Go+n*3);
    cblas_dcopy(3,Go+n*3,1,U+n*3,1);
  }

  double *Unew = (double *)mkl_malloc(3 * (N + 1) * sizeof(double), 64);
  for (int k=0;k<K;k++) {
    double* Fn=(double*)mkl_calloc((N+1)*3,sizeof(double),64);
    tbb::parallel_for(0,N,[&TT,&U,&F,&Fn](int n) {
      SForwardEuler(TT[n],TT[n+1],U+n*3,3,F,Fn+(n+1)*3);
    });
    cblas_dcopy(3,u0,1,Unew,1);
    double* Gn=(double*)mkl_calloc((N+1)*3,sizeof(double),64);
    for (int n=1;n<=N;n++) {
      SForwardEuler(TT[n-1],TT[n],Unew+(n-1)*3,3,G,Gn+n*3);
      for (int ii=0;ii<3;ii++) {
        (Unew+n*3)[ii]=(Fn+n*3)[ii]+(Gn+n*3)[ii]-(Go+n*3)[ii];
      }
    }
    cblas_dcopy((N+1)*3,Gn,1,Go,1);
    cblas_dcopy((N+1)*3,Unew,1,U,1);
    mkl_free(Gn);
    mkl_free(Fn);
  }


  mkl_free(Unew);
  mkl_free(Go);
  mkl_free(TT);
};

int main() {
  const double T = 5;
  const double u0[3] = {20, 5, -5};
  const int K = 20;
  const int N = 500;
  const int MF=10;
  const int MG=1;


  double *RES = (double *)mkl_malloc((N + 1) * 3 * sizeof(double), 64);

  Parareal(MF,MG,0,T,u0,N,K,RES);

  mkl_free(RES);
  return 0;
}