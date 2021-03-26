// Title: laplace.cpp
// Description: Program to solve the Laplace equation in two dimensions with the Jacobi method.
// Author: Kate Storey-Fisher
// Date: 2021-03-26

#include <stdio.h>
#include <cmath>
#include <omp.h>


void jacobi_serial(double* u, double* f, double*A, long N, int maxiter){
  // assume lexicographic ordering, i row-major order; access [i,j] with [i*N+j)
  double hh, diff_norm, diff_thresh;
  const long N2 = N+2;
  double* u_prev = (double*) malloc(N2*N2 * sizeof(double));
  hh = 1.0/(double)((N+1)*(N+1)); //h times h

  for (int k=0; k<maxiter; k++) {
    // copy elements of u array into u_prev
    for (long ii = 0; ii < N2*N2; ii++) u_prev[ii] = u[ii];
    // exclude padding
    for (int i=1; i<N+1; i++) {
      for (int j=1; j<N+1; j++) {
        // overwrite u with the kth iteration, based on the previous u
        //printf("f %f\n", f[i*N2+j]);
        //printf("uprev %f\n", u_prev[(i-1)*N2+j]);
        //printf("%f\n", hh*f[i*N2+j]);
        u[i*N2+j] = 0.25*(-hh*f[i*N2+j] + u_prev[(i-1)*N2 + j] + u_prev[i*N2+(j-1)] + u_prev[(i+1)*N2+j] + u_prev[i*N2+(j+1)]);
        //printf("u_prev[00]=%f, u[00]=%f\n", u_prev[i*N+j], u[i*N+j]);
        //printf("%f ", u[i*N2+j]);
        //printf("\n");
      }
      //printf("\n");
    }

    diff_norm = 0;
    for (int i=1; i<N+1; i++) {
      for (int j=1; j<N+1; j++) {
        diff_norm += (u[i*N2+j] - u_prev[i*N2+j])*(u[i*N2+j] - u_prev[i*N2+j]);
      }
    }
    diff_norm = sqrt(diff_norm);

    // define threshhold as decreasing the diffnorm by 10^6 times the initial
    if (k==0){
      diff_thresh = 1e-6 * diff_norm;
    }
    printf("k=%d, diff_norm=%e (diff_thresh=%e)\n", k, diff_norm, diff_thresh);
    if (diff_norm < diff_thresh) {
      printf("Threshold reached after %d iterations! diff_norm=%e\n", k, diff_norm);
      break;
    }
  }

  free(u_prev);
}


int main(int argc, char** argv) {
  
  const long N = 7;
  const long N2 = N+2;
  int maxiter = 5000;

  // Initialize f with all 1s; pad edges
  double* f = (double*) malloc(N2*N2 * sizeof(double));
  for (long ij = 0; ij < N2*N2; ij++) f[ij] = 1;

  // Initialize u with all 0s; pad edges
  //double* u_prev = (double*) malloc(N*N * sizeof(double));
  double* u = (double*) malloc(N2*N2 * sizeof(double));
  for (long ij = 0; ij < N2*N2; ij++) u[ij] = 0;

  // Initialize A (we will ignore edges later)
  double* A = (double*) malloc(N2*N2 * sizeof(double));
  for (long ij = 0; ij < N2*N2; ij++) A[ij] = 0; // start with zeros everywhere
  double prefac = (N+1)*(N+1); // 1/h^2, h = 1/(N+1)
  for (long i = 1; i < N+1; i++) {
    A[i*N2+i] = 2*prefac; // diagonal gets 2s
    if (i==N-1) continue;
    A[i*N2+(i+1)] = -prefac; // edge above diag gets -1s
    A[(i+1)*N2+i] = -prefac; // edge below diag gets -1s
  }

  // Iteratively approximate solution
  double tt = omp_get_wtime();
  jacobi_serial(u, f, A, N, maxiter);
  printf("jacobi serial = %fs\n", omp_get_wtime() - tt);
    
  free(u);
  //free(u_prev);
  free(f);
  free(A);

  return 0;
}

