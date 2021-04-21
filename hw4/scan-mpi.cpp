#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_mpi(long* prefix_sum, const long* A, long n) {
  // TODO: implement multi-threaded OpenMP scan
  if (n == 0) return;
 
  int rank, p;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
   
  // Set up variables
  long chunksize;
  chunksize = n/p;
  //chunksize = ceil((double)n/(double)p); // will have to deal with potential for last chunk to be smaller
  long offsets[p];
  //offsets[0] = 0;

  float *prefix_sum_chunk = malloc(sizeof(long) * chunksize);
  // 0 is rank of root, the sending process
  MPI_Scatter(A, chunksize, MPI_LONG, prefix_sum_chunk, chunksize, MPI_LONG, 0, comm);

  // Have each thread compute its part of prefix_sum
  if (rank==0){
      prefix_sum[0] = 0;
  }
  else{
      prefix_sum[start] = A[start-1];
  }

  for (long i=0; i<chunksize; i++){
      prefix_sum_chunk[i] = prefix_sum_chunk[i-1] + A[i-1];
  }
  long offset = prefix_sum_chunk[0];
  printf("Thread %d computed its chunk\n", tid);
  // all threads have finished their chunk
  
  MPI_Allgather(&offset, 1, MPI_LONG, offsets, 1, MPI_LONG, comm);
  
  // Compute the needed offsets
  printf("Adjusting for constant offset\n");
  long offset_total;
  for (int t=0; t<rank; t++){
      offset_total += offsets[t];
  }
  // Add the offsets back to prefix_sum, in parallel again
  for (long i=0; i<chunksize; i++){
      prefix_sum_chunk[i] += offset_total;
  }
  
  // Back into main array
  // TODO: check
  MPI_Gather(prefix_sum_chunk, chunksize, MPI_LONG, prefix_sum, n, MPI_LONG, 0, comm);
    
  printf("Done!\n");
}

int main() {

  long N = 8;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);


  int rank, p;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  if (N%p != 0){
      printf("N must be divisible by the number of processors p!\n");
      return -1;
  }

  /* get name of host running MPI process */
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Rank %d/%d running on %s.\n", rank, p, processor_name);

  tt = omp_get_wtime();
  scan_mpi(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
