// Title: laplace.cpp
// Description: Program to solve the Laplace equation in one dimension with the Jacobi method.
// Author: Kate Storey-Fisher
// Date: 2021-02-18
// Run: $ g++ -O3 -std=c++11 laplace.cpp && ./a.out

#include <stdio.h>
#include <cmath>
#include "utils.h"

int main(int argc, char** argv) {
  
  const long N = 10000;
  int maxiter = 5000;

  // Initialize f with all 1s
  double* f = (double*) malloc(N * sizeof(double));
  for (long i = 0; i < N; i++) f[i] = 1;

  // Initialize u with all 0s, and setup u_prev
  double* u_prev = (double*) malloc(N * sizeof(double));
  double* u = (double*) malloc(N * sizeof(double));
  for (long i = 0; i < N; i++) u[i] = 0;

  // Initialize A
  double* A = (double*) malloc(N*N * sizeof(double));
  for (long ij = 0; ij < N*N; ij++) A[ij] = 0; // start with zeros everywhere
  double prefac = (N+1)*(N+1); // 1/h^2, h = 1/(N+1)
  for (long i = 0; i < N; i++) {
    A[i+i*N] = 2*prefac; // diagonal gets 2s
    if (i==N-1) continue;
    A[i+(i+1)*N] = -prefac; // edge above diag gets -1s
    A[(i+1)+i*N] = -prefac; // edge below diag gets -1s
  }

  // Iteratively approximate solution
  Timer t;
  t.tic();
  double sum, Au_i, Auf, residual_norm, residual_thresh;
  for (int k=0; k<maxiter; k++) { 
    // copy elements of u array into u_prev
    for (long ii = 0; ii < N; ii++) u_prev[ii] = u[ii];
    for (int i=0; i<N; i++) {
      sum = 0;
      // the only terms of A we need are these, so avoid another loop; 
      // the u values are indexed by u[j]
      if (i>0) sum += A[i+(i-1)*N] * u_prev[i-1];
      if (i<N-1) sum += A[i+(i+1)*N] * u_prev[i+1];
      // overwrite u with the kth iteration, based on the previous u
      u[i] = 1/A[i+i*N] * (f[i] - sum);
    }

    // Compute the norm of the residual between our solution and the truth
    residual_norm = 0;
    for (int i=0; i<N; i++) {
      Au_i = 0;
      // Only these elements of Au are nonzero so avoid another loop
      Au_i += A[i+i*N] * u[i];
      if (i>0) Au_i += A[i+(i-1)*N] * u[i-1];
      if (i<N-1) Au_i += A[i+(i+1)*N] * u[i+1];
      Auf = Au_i - f[i];
      residual_norm += Auf*Auf; // sum the square differences
    }
    residual_norm = sqrt(residual_norm); // take square root
    // define threshhold as decreasing the residual by 10^6 times the initial
    if (k==0) {
        residual_thresh = 1e-6 * residual_norm; 
    }
    printf("k = %d, residual_norm = %e (r_norm/r_thresh=%e)\n", k, residual_norm, residual_norm/(1e6*residual_thresh));
    if (residual_norm < residual_thresh) {
        printf("Threshold reached after %d iterations! Residual: %f\n", k, residual_norm);
        break;
    }
  }
  double time = t.toc();
  printf("Time: %f\n", time);
    
  free(u);
  free(u_prev);
  free(f);
  free(A);

  return 0;
}

