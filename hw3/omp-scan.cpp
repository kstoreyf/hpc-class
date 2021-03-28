#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
    //printf("pcorr[%ld] = %ld\n", i, prefix_sum[i]);
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  // TODO: implement multi-threaded OpenMP scan
  if (n == 0) return;
  
  // Set up variables
  int tid, p;
  long chunksize;
  p = omp_get_max_threads();
  printf("Num threads: %d\n", p);
  chunksize = ceil((double)n/(double)p); // will have to deal with potential for last chunk to be smaller
  long offsets[p];
  offsets[0] = 0;

  // Have each thread compute its part of prefix_sum
  #pragma omp parallel private(tid) shared(prefix_sum, A, n, chunksize)
  {
  tid = omp_get_thread_num();
  long start, stop;
  start = tid*chunksize;
  stop = std::min(start + chunksize, n); //in case n not divisible by p. used ceil, so min
  if (tid==0){
      prefix_sum[0] = 0;
  }
  else{
      prefix_sum[start] = A[start-1];
  }
  for (long i=start+1; i<stop; i++){
      prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
  printf("Thread %d computed its chunk\n", tid);
  } //end pragma region
  // all threads have finished their chunk
  
  // Compute the needed offsets
  printf("Adjusting for constant offset\n");
  int t;
  long const_index;
  for (int t=1; t<p; t++){
      const_index = std::min(t*chunksize, n)-1;
      offsets[t] = offsets[t-1] + prefix_sum[const_index];
  }

  // Add the offsets back to prefix_sum, in parallel again
  #pragma omp parallel private(tid) shared(prefix_sum, A, n, chunksize, offsets)
  {
  tid = omp_get_thread_num();
  long start, stop;
  start = tid*chunksize;
  stop = std::min(start + chunksize, n);
  for (long i=start; i<stop; i++){
      prefix_sum[i] += offsets[tid];
  }
  } //end pragma region
    
  printf("Done!\n");

}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
