// Title: laplace.cpp
// Description: Program to solve the Laplace equation in two dimensions with the Jacobi method.
// Author: Kate Storey-Fisher
// Date: 2021-03-26

#include <stdio.h>
#include <cmath>
#ifdef OPENMP
#include <omp.h>
#endif


void gauss_seidel(double* u, double* f, long N, int maxiter){
  // assume lexicographic ordering, i row-major order; access [i,j] with [i*N+j)
  double hh, diff_norm, diff_thresh, u_prev;
  const long N2 = N+2;
  hh = 1.0/(double)((N+1)*(N+1)); //h times h

  for (int k=0; k<maxiter; k++) {
    // copy elements of u array into u_prev
    // exclude padding
    diff_norm = 0;
    for (long i=1; i<N+1; i++) {
      for (long j=1; j<N+1; j++) {
        u_prev = u[i*N2+j];
        u[i*N2+j] = 0.25*(hh*f[i*N2+j] + u[(i-1)*N2 + j] + u[i*N2+(j-1)] + u[(i+1)*N2+j] + u[i*N2+(j+1)]);
        diff_norm += (u[i*N2+j] - u_prev)*(u[i*N2+j] - u_prev);
      }
    }
    diff_norm = sqrt(diff_norm);

    // define threshhold as decreasing the diffnorm by 10^6 times the initial
    if (k==0){
      diff_thresh = 1e-6 * diff_norm;
    }
    if (diff_norm < diff_thresh) {
      printf("Threshold reached after %d iterations! diff_norm=%e\n", k, diff_norm);
      break;
    }
    if (k==maxiter-1) printf("Maxiter reached (%d iterations)! diff_norm=%e\n", maxiter, diff_norm);
  }

}


void gauss_seidel_colored(double* u, double* f, long N, int maxiter){
  // assume lexicographic ordering, i row-major order; access [i,j] with [i*N+j)
  double hh, diff_norm, diff_thresh, u_prev;
  long jstart, istart;
  const long N2 = N+2;
  hh = 1.0/(double)((N+1)*(N+1)); //h times h

  for (int k=0; k<maxiter; k++) {
    // exclude padding
    diff_norm = 0;
    //update all red points (i+j even)
    for (long i=1; i<N+1; i++) {
      if (i%2==1) jstart = 1;
      else jstart = 2;
      for (long j=jstart; j<N+1; j+=2) {
        u_prev = u[i*N2+j];
        u[i*N2+j] = 0.25*(hh*f[i*N2+j] + u[(i-1)*N2 + j] + u[i*N2+(j-1)] + u[(i+1)*N2+j] + u[i*N2+(j+1)]);
        diff_norm += (u[i*N2+j] - u_prev)*(u[i*N2+j] - u_prev);
      }
    }
    //update all black points (i+j odd)
    for (long i=1; i<N+1; i++) {
      if (i%2==1) jstart = 2;
      else jstart = 1;
      for (long j=jstart; j<N+1; j+=2) {
        u_prev = u[i*N2+j];
        u[i*N2+j] = 0.25*(hh*f[i*N2+j] + u[(i-1)*N2 + j] + u[i*N2+(j-1)] + u[(i+1)*N2+j] + u[i*N2+(j+1)]);
        diff_norm += (u[i*N2+j] - u_prev)*(u[i*N2+j] - u_prev);
      }
    }
    diff_norm = sqrt(diff_norm);

    // define threshhold as decreasing the diffnorm by 10^6 times the initial
    if (k==0){
      diff_thresh = 1e-6 * diff_norm;
    }
    //printf("k=%d, diff_norm=%e (diff_thresh=%e)\n", k, diff_norm, diff_thresh);
    if (diff_norm < diff_thresh) {
      printf("Threshold reached after %d iterations! diff_norm=%e\n", k, diff_norm);
      break;
    }
    if (k==maxiter-1) printf("Maxiter reached (%d iterations)! diff_norm=%e\n", maxiter, diff_norm);
  }

}


void gauss_seidel_colored_parallel(double* u, double* f, long N, int maxiter){
  // assume lexicographic ordering, i row-major order; access [i,j] with [i*N+j)
  double hh, diff_norm, diff_thresh, u_prev;
  long jstart, istart;
  const long N2 = N+2;
  hh = 1.0/(double)((N+1)*(N+1)); //h times h
  #ifdef OPENMP
  printf("Num threads: %d\n", omp_get_max_threads());
  #endif

  for (int k=0; k<maxiter; k++) {
    // exclude padding
    diff_norm = 0;
    //update all red points (i+j even)
    #ifdef OPENMP
    #pragma omp parallel for schedule(static) reduction(+:diff_norm) shared(u) private(u_prev)
    #endif
    for (long i=1; i<N+1; i++) {
      if (i%2==1) jstart = 1;
      else jstart = 2;
      for (long j=jstart; j<N+1; j+=2) {
        u_prev = u[i*N2+j];
        u[i*N2+j] = 0.25*(hh*f[i*N2+j] + u[(i-1)*N2 + j] + u[i*N2+(j-1)] + u[(i+1)*N2+j] + u[i*N2+(j+1)]);
        diff_norm += (u[i*N2+j] - u_prev)*(u[i*N2+j] - u_prev);
      }
    }
    //update all black points (i+j odd)
    #ifdef OPENMP
    #pragma omp parallel for schedule(static) reduction(+:diff_norm) shared(u) private(u_prev)
    #endif
    for (long i=1; i<N+1; i++) {
      if (i%2==1) jstart = 2;
      else jstart = 1;
      for (long j=jstart; j<N+1; j+=2) {
        u_prev = u[i*N2+j];
        u[i*N2+j] = 0.25*(hh*f[i*N2+j] + u[(i-1)*N2 + j] + u[i*N2+(j-1)] + u[(i+1)*N2+j] + u[i*N2+(j+1)]);
        diff_norm += (u[i*N2+j] - u_prev)*(u[i*N2+j] - u_prev);
      }
    }
    diff_norm = sqrt(diff_norm);

    // define threshhold as decreasing the diffnorm by 10^6 times the initial
    if (k==0){
      diff_thresh = 1e-6 * diff_norm;
    }
    //printf("k=%d, diff_norm=%e (diff_thresh=%e)\n", k, diff_norm, diff_thresh);
    if (diff_norm < diff_thresh) {
      printf("Threshold reached after %d iterations! diff_norm=%e\n", k, diff_norm);
      break;
    }
    if (k==maxiter-1) printf("Maxiter reached (%d iterations)! diff_norm=%e\n", maxiter, diff_norm);
  }

}


int main(int argc, char** argv) {
  
  const long N = 512;
  const long N2 = N+2;
  int maxiter = 1000;

  // Initialize f with all 1s; pad edges
  double* f = (double*) malloc(N2*N2 * sizeof(double));
  for (long ij = 0; ij < N2*N2; ij++) f[ij] = 1.0;

  // Initialize u with all 0s; pad edges
  double* u = (double*) malloc(N2*N2 * sizeof(double));
  double* u_col = (double*) malloc(N2*N2 * sizeof(double));
  double* u_par = (double*) malloc(N2*N2 * sizeof(double));
  for (long ij = 0; ij < N2*N2; ij++){
       u[ij] = 0.0;
       u_col[ij] = 0.0;
       u_par[ij] = 0.0;
  }

  // Iteratively approximate solution
  #ifdef OPENMP
  double tt = omp_get_wtime();
  #endif
  gauss_seidel(u, f, N, maxiter);
  #ifdef OPENMP
  printf("gauss-seidel (serial) = %fs\n", omp_get_wtime() - tt);
  #endif
  
  #ifdef OPENMP 
  tt = omp_get_wtime();
  #endif
  gauss_seidel_colored(u_col, f, N, maxiter);
  #ifdef OPENMP
  printf("gauss-seidel colored (serial) = %fs\n", omp_get_wtime() - tt);
  #endif

  #ifdef OPENMP
  tt = omp_get_wtime();
  #endif
  gauss_seidel_colored_parallel(u_par, f, N, maxiter);
  #ifdef OPENMP
  printf("gauss-seidel colored (parallel) = %fs\n", omp_get_wtime() - tt);
  #endif
   
  free(u);
  free(u_col);
  free(u_par);
  free(f);

  return 0;
}

