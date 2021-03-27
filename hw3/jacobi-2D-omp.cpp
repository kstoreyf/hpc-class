// Title: laplace.cpp
// Description: Program to solve the Laplace equation in two dimensions with the Jacobi method.
// Author: Kate Storey-Fisher
// Date: 2021-03-26

#include <stdio.h>
#include <cmath>
#include <omp.h>


void jacobi_serial(double* u, double* f, long N, int maxiter){
  // assume lexicographic ordering, i row-major order; access [i,j] with [i*N+j)
  double hh, diff_norm, diff_thresh;
  const long N2 = N+2;
  double* u_prev = (double*) malloc(N2*N2 * sizeof(double));
  hh = 1.0/(double)((N+1)*(N+1)); //h times h

  for (int k=0; k<maxiter; k++) {
    // copy elements of u array into u_prev
    for (long ij = 0; ij < N2*N2; ij++) u_prev[ij] = u[ij];
    // exclude padding
    diff_norm = 0;
    for (long i=1; i<N+1; i++) {
      for (long j=1; j<N+1; j++) {
        // overwrite u with the kth iteration, based on the previous u
        //printf("f %f\n", f[i*N2+j]);
        //printf("uprev %f\n", u_prev[(i-1)*N2+j]);
        //printf("%f\n", hh*f[i*N2+j]);
        u[i*N2+j] = 0.25*(hh*f[i*N2+j] + u_prev[(i-1)*N2 + j] + u_prev[i*N2+(j-1)] + u_prev[(i+1)*N2+j] + u_prev[i*N2+(j+1)]);
        diff_norm += (u[i*N2+j] - u_prev[i*N2+j])*(u[i*N2+j] - u_prev[i*N2+j]);
        //printf("u_prev[00]=%f, u[00]=%f\n", u_prev[i*N+j], u[i*N+j]);
        //printf("%f ", u[i*N2+j]);
        //printf("\n");
      }
      //printf("\n");
    }

    //diff_norm = 0;
    //for (long i=1; i<N+1; i++) {
    //  for (long j=1; j<N+1; j++) {
    //    diff_norm += (u[i*N2+j] - u_prev[i*N2+j])*(u[i*N2+j] - u_prev[i*N2+j]);
    //  }
    //}
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
    if (k==maxiter-1) printf("Maxiter reached (%d iterations)! diff_norm=%e\n", k, diff_norm);
  }

  free(u_prev);
}


void jacobi_parallel(double* u, double* f, long N, int maxiter){
  // assume lexicographic ordering, i row-major order; access [i,j] with [i*N+j)
  double hh, diff_norm, diff_thresh;
  const long N2 = N+2;
  double* u_prev_par = (double*) malloc(N2*N2 * sizeof(double));
  hh = 1.0/(double)((N+1)*(N+1)); //h times h

  for (int k=0; k<maxiter; k++) {
    // copy elements of u array into u_prev
    for (long ij = 0; ij < N2*N2; ij++) u_prev_par[ij] = u[ij];
    // exclude padding
    diff_norm = 0;
    #pragma omp parallel for schedule(static) reduction(+:diff_norm)
    for (long i=1; i<N+1; i++) {
      for (long j=1; j<N+1; j++) {
        // overwrite u with the kth iteration, based on the previous u
        //printf("f %f\n", f[i*N2+j]);
        //printf("uprev %f\n", u_prev[(i-1)*N2+j]);
        //printf("%f\n", hh*f[i*N2+j]);
        //printf("Thread %d computing u[%ld,%ld]\n", omp_get_thread_num(), i, j);
        u[i*N2+j] = 0.25*(hh*f[i*N2+j] + u_prev_par[(i-1)*N2 + j] + u_prev_par[i*N2+(j-1)] + u_prev_par[(i+1)*N2+j] + u_prev_par[i*N2+(j+1)]);
        diff_norm += (u[i*N2+j] - u_prev_par[i*N2+j])*(u[i*N2+j] - u_prev_par[i*N2+j]);
        //printf("u_prev[00]=%f, u[00]=%f\n", u_prev[i*N+j], u[i*N+j]);
        //printf("%f ", u[i*N2+j]);
        //printf("\n");
      }
      //printf("\n");
    }

    //diff_norm = 0;
    //#pragma omp parallel for reduction(+:diff_norm)
    //for (long i=1; i<N+1; i++) {
    //  for (long j=1; j<N+1; j++) {
    //    diff_norm += (u[i*N2+j] - u_prev_par[i*N2+j])*(u[i*N2+j] - u_prev_par[i*N2+j]);
    //  }
    //}
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
    if (k==maxiter-1) printf("Maxiter reached (%d iterations)! diff_norm=%e\n", k, diff_norm);
  }

  free(u_prev_par);
}


int main(int argc, char** argv) {
  
  const long N = 100;
  const long N2 = N+2;
  int maxiter = 1000;

  // Initialize f with all 1s; pad edges
  double* f = (double*) malloc(N2*N2 * sizeof(double));
  for (long ij = 0; ij < N2*N2; ij++) f[ij] = 1;

  // Initialize u with all 0s; pad edges
  double* u = (double*) malloc(N2*N2 * sizeof(double));
  double* u_par = (double*) malloc(N2*N2 * sizeof(double));
  for (long ij = 0; ij < N2*N2; ij++) u[ij] = 0;
  for (long ij = 0; ij < N2*N2; ij++) u_par[ij] = 0;

  // Iteratively approximate solution
  double tt = omp_get_wtime();
  jacobi_serial(u, f, N, maxiter);
  printf("jacobi serial = %fs\n", omp_get_wtime() - tt);
  
  tt = omp_get_wtime();
  jacobi_parallel(u_par, f, N, maxiter);
  printf("jacobi parallel = %fs\n", omp_get_wtime() - tt);
   
  free(u);
  free(u_par);
  free(f);

  return 0;
}

