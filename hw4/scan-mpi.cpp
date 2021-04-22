#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
double scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return 0;

  double tt = MPI_Wtime();
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
  tt = MPI_Wtime() - tt;
  return tt;
}

double scan_mpi(long* prefix_sum, const long* A, long n, MPI_Comm comm) {
  if (n == 0) return 0;

  // Set up MPI 
  int rank, p;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  // Set up other variables
  long chunksize;
  chunksize = n/p;
  long offsets[p];

  long *A_chunk = (long*) malloc(sizeof(long) * chunksize);
  long *prefix_sum_chunk = (long*) malloc(sizeof(long) * chunksize);

  MPI_Barrier(comm);
  double tt = MPI_Wtime();
  // 0 is rank of root, the sending process
  MPI_Scatter(A, chunksize, MPI_LONG, A_chunk, chunksize, MPI_LONG, 0, comm);

  // Have each thread compute its part of prefix_sum
  prefix_sum_chunk[0] = 0;
  for (long i=1; i<chunksize; i++){
      prefix_sum_chunk[i] = prefix_sum_chunk[i-1] + A_chunk[i-1];
      //printf("P%d, prefix_sum_chunk[%d]=%ld\n", rank, i, prefix_sum_chunk[i]);
  }
  // For prefix_sum (as opposed to scan), need the final term bc otherwise
  // never access that value of A
  long offset = prefix_sum_chunk[chunksize-1] + A_chunk[chunksize-1];
  printf("Process %d computed its chunk, found offset %ld\n", rank, offset);
  // all threads have finished their chunk
  
  MPI_Allgather(&offset, 1, MPI_LONG, offsets, 1, MPI_LONG, comm);
  
  // Compute the needed offsets
  long offset_total = 0;
  for (int t=0; t<rank; t++){
      offset_total += offsets[t];
  }
  // Add the offsets back to prefix_sum
  for (long i=0; i<chunksize; i++){
      prefix_sum_chunk[i] += offset_total;
  }
  
  // Back into main array
  MPI_Gather(prefix_sum_chunk, chunksize, MPI_LONG, prefix_sum, chunksize, MPI_LONG, 0, comm);
    
  tt = MPI_Wtime() - tt;
  return tt;
}

int main(int argc, char *argv[]) {

  MPI_Init(&argc, &argv);
  int rank, p;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  long N = 10000000;
  if (N%p != 0){
      printf("N must be divisible by the number of processors p!\n");
      return -1;
  }

  long* A;
  long* B0;
  long* B1;
  if (rank==0){
    A = (long*) malloc(N * sizeof(long));
    B0 = (long*) malloc(N * sizeof(long));
    B1 = (long*) malloc(N * sizeof(long));
    for (long i = 0; i < N; i++) A[i] = rand();
  }

  double tt;
  if(rank==0){
    tt = scan_seq(B0, A, N);
    printf("sequential-scan = %fs\n", tt);
  }

  /* get name of host running MPI process */
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Rank %d/%d running on %s.\n", rank, p, processor_name);

  tt = scan_mpi(B1, A, N, comm);

  if (rank==0) {

    printf("parallel-scan   = %fs\n", tt);
    long err = 0;
    for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
    printf("error = %ld\n", err);
  
    free(A);
    free(B0);
    free(B1);
  }

  MPI_Finalize();
  return 0;
}
