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
  //for (long i=0; i<n; i++){
  //  printf("A[%ld] = %ld\n", i, A[i]);
  //}
  int tid, p;
  long chunksize;
  p = omp_get_max_threads();
  printf("Num threads: %d\n", p);
  chunksize = ceil((double)n/(double)p); // will have to deal with potential for last chunk to be smaller
  //prefix_sum[0] = 0; //shouldnt this be A[0]?? also related, in above, never get to A[n-1]
  #pragma omp parallel private(tid) shared(prefix_sum, A, n, chunksize)
  {
  tid = omp_get_thread_num();
  //p =  omp_get_num_threads();
  long start, stop;
  start = tid*chunksize;
  stop = std::min(start + chunksize, n); //in case n not divisible by p. used ceil, so min
  //printf("n %ld, p %d\n", n, p);
  //printf("chunksize=%ld, start=%ld, stop=%ld\n", chunksize, start, stop);
  if (tid==0){
      prefix_sum[0] = 0;
  }
  else{
      prefix_sum[start] = A[start-1];
  }
  printf("Set up chunks\n");
  // each will have its own prefix sum
  
  //prefix_sum[start] = A[start]; // should be this if serial is how i think it should be!
  for (long i=start+1; i<stop; i++){
      prefix_sum[i] = prefix_sum[i-1] + A[i-1];
      //printf("p[%ld] = %ld\n", i, prefix_sum[i]);
  }
  printf("Thread %d computed its chunk\n", tid);
  //#pragma omp barrier
  } //end pragma region
  // all threads have finished their chunk
   
  //now correct for the constant 
  printf("Adjusting for constant offset\n");
  int t;
  long const_index;
  for (long i=0; i<n; i++){
      // sum over the contants from chunks below yours
      // figure out what chunk t this part was in
      t = floor(i/chunksize); //should be equiv to orig tid
      // the constant to add is the value before "stop";
      // only need to add one bc we will have already
      // fixed all the earlier values
      if (t>0){
        const_index = std::min(t*chunksize, n)-1;
        prefix_sum[i] += prefix_sum[const_index];
      }
      //printf("pfinal[%ld] = %ld\n", i, prefix_sum[i]);
  }
  //} //end pragma region
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
