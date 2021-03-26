// Title: laplace.cpp
// Description: Program to solve the Laplace equation in two dimensions with the Jacobi method.
// Author: Kate Storey-Fisher
// Date: 2021-03-26

#include <stdio.h>
#include <cmath>
#include <omp.h>


void jacobi_serial(double* u, double* f, double*A, long N, int maxiter){
  // assume lexicographic ordering, i row-major order; access [i,j] with [i*N+j)
  double hh, resid, residual_norm, residual_thresh;
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

    double diffnorm = 0;
    for (int i=1; i<N+1; i++) {
      for (int j=1; j<N+1; j++) {
        diffnorm += (u[i*N2+j] - u_prev[i*N2+j])*(u[i*N2+j] - u_prev[i*N2+j]);
      }
    }
    diffnorm = sqrt(diffnorm);
    printf("diffnorm: %f\n", diffnorm);

    double* f_approx = (double*) malloc(N2*N2 * sizeof(double));
    for (long i = 0; i < N2*N2; i++) f_approx[i] = 0;
    residual_norm = 0;
    for (int i=1; i<N+1; i++) {
      for (int j=1; j<N+1; j++) {
        // Only these elements of Au are nonzero
        // this is for a column of a, and row of u (swap the indices)
        if (i==j || i==j-1 || i==j+1){
            //printf("i=%d, j=%d\n", i, j);
            f_approx[i*N2+j] += A[i*N2+j] * u[j*N2+i];
        }

        //if (i>0) Au_i += A[i*N+(i-1)] * u[(i-1)*N+i]; // for a row i, the j values are above & below
        //if (i<N-1) Au_i += A[i*N+(i+1)] * u[(i+1)*N+i];
      }
    }
    for (int i=1; i<N+1; i++) {
      for (int j=1; j<N+1; j++) {
        resid = f_approx[i*N2+j] - f[i*N2+j];
        residual_norm += resid*resid;
      }
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

  free(u_prev);
}


int main(int argc, char** argv) {
  
  const long N = 7;
  const long N2 = N+2;
  int maxiter = 100;

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

