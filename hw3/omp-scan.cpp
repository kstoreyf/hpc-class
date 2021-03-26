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
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  // TODO: implement multi-threaded OpenMP scan
  if (n == 0) return;
  int tid, p;
  //prefix_sum[0] = 0; //shouldnt this be A[0]?? also related, in above, never get to A[n-1]
  #pragma omp parallel private(tid)
  {
  tid = omp_get_thread_num();
  p =  omp_get_num_threads();
  long chunksize, start, stop;
  chunksize = ceil(n/p); // will have to deal with potential for last chunk to be smaller
  if (tid==0){
      printf("Num threads: %d\n", p);
      prefix_sum[0] = 0;
  }
  else{
      prefix_sum[start] = A[start-1];
  }
  printf("Set up chunks\n");
  // each will have its own prefix sum
  start = p*chunksize;
  stop = std::min(start + chunksize, n); //in case n not divisible by p. used ceil, so min
  
  //prefix_sum[start] = A[start]; // should be this if serial is how i think it should be!
  for (long i=start+1; i<stop; i++){
      prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
  printf("Thread %d computed its chunk\n", tid);
  #pragma omp barrier
  //} //end pragma region
  // all threads have finished their chunk
  
  //now correct for the constant 
  // doing this because we still need chunksize and p
  if (tid==0){
      printf("Adjusting for constant offset\n");
      int t;
      long const_index;
      for (long i=0; i<n; i++){
          // sum over the contants from chunks below yours
          // what chunk was this part in?
          t = floor(i/chunksize); //should be equiv to tid
          for(int l=0; l<t; l++){
              // the constant to add is the value before "stop"
              const_index = std::min((p+1)*chunksize, n)-1;
              prefix_sum[i] += prefix_sum[const_index];
          }
      }
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
