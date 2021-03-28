// Title: laplace.cpp
// Description: Program to solve the Laplace equation in two dimensions with the Jacobi method.
// Author: Kate Storey-Fisher
// Date: 2021-03-26

#include <stdio.h>
#include <cmath>
#ifdef OPENMP
    #include <omp.h>
#endif


void jacobi_serial(double* u, double* f, long N, int maxiter){
  // assume lexicographic ordering, i row-major order; access [i,j] with [i*N+j)
  double hh, diff_norm, diff_thresh;
  const long N2 = N+2;
  double* u_prev = (double*) malloc(N2*N2 * sizeof(double));
  hh = 1.0/(double)((N+1)*(N+1)); //h times h

  // Loop over iterations
  for (int k=0; k<maxiter; k++) {
    // copy elements of u array into u_prev
    for (long ij = 0; ij < N2*N2; ij++) u_prev[ij] = u[ij];
    // exclude padding
    diff_norm = 0;

    // update u values based on previous u, and compute 
    // the difference to check convergence
    for (long i=1; i<N+1; i++) {
      for (long j=1; j<N+1; j++) {
        u[i*N2+j] = 0.25*(hh*f[i*N2+j] + u_prev[(i-1)*N2 + j] + u_prev[i*N2+(j-1)] + u_prev[(i+1)*N2+j] + u_prev[i*N2+(j+1)]);
        diff_norm += (u[i*N2+j] - u_prev[i*N2+j])*(u[i*N2+j] - u_prev[i*N2+j]);
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
    if (k==maxiter-1) printf("Maxiter reached (%d iterations)! diff_norm=%e\n", k, diff_norm);
  }

  free(u_prev);
}


void jacobi_parallel(double* u, double* f, long N, int maxiter){
  // assume lexicographic ordering, i row-major order; access [i,j] with [i*N+j)
  double hh, diff_norm, diff_thresh;
  const long N2 = N+2;
  double* u_prev = (double*) malloc(N2*N2 * sizeof(double));
  hh = 1.0/(double)((N+1)*(N+1)); //h times h
  #ifdef OPENMP
  printf("Num threads: %d\n", omp_get_max_threads());
  #endif
  // loop over iterations
  for (int k=0; k<maxiter; k++) {
    // copy elements of u array into u_prev
    #ifdef OPENMP
    #pragma omp parallel for schedule(static) shared(u, u_prev)
    #endif
    for (long ij = 0; ij < N2*N2; ij++) u_prev[ij] = u[ij];
    // exclude padding
    diff_norm = 0;

    // We parallelize this with a basic scheduled for loop, so that 
    // each thread will take a chunk of rows; this is fine because 
    // the only dependence is on the previous u, not what the other
    // threads are doing. 
    // We must make sure to perform a reduction of diff_norm to 
    // keep track of convergence.
    #ifdef OPENMP
    #pragma omp parallel for schedule(static) reduction(+:diff_norm) shared(u, u_prev)
    #endif
    for (long i=1; i<N+1; i++) {
      for (long j=1; j<N+1; j++) {
        // overwrite u with the kth iteration, based on the previous u
        u[i*N2+j] = 0.25*(hh*f[i*N2+j] + u_prev[(i-1)*N2 + j] + u_prev[i*N2+(j-1)] + u_prev[(i+1)*N2+j] + u_prev[i*N2+(j+1)]);
        diff_norm += (u[i*N2+j] - u_prev[i*N2+j])*(u[i*N2+j] - u_prev[i*N2+j]);
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
    if (k==maxiter-1) printf("Maxiter reached (%d iterations)! diff_norm=%e\n", k, diff_norm);
  }

  free(u_prev);
}


int main(int argc, char** argv) {
  
  const long N = 1000;
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
  #ifdef OPENMP
  double tt = omp_get_wtime();
  #endif
  jacobi_serial(u, f, N, maxiter);
  #ifdef OPENMP
  printf("jacobi serial = %fs\n", omp_get_wtime() - tt);
  #endif
  
  // Parallel solution
  #ifdef OPENMP
  tt = omp_get_wtime();
  #endif
  jacobi_parallel(u_par, f, N, maxiter);
  #ifdef OPENMP
  printf("jacobi parallel = %fs\n", omp_get_wtime() - tt);
  #endif
   
  free(u);
  free(u_par);
  free(f);

  return 0;
}

