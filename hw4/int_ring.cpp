/* Run with: mpicxx  -std=c++11 -O3 -march=native -fopenmp    int_ring.cpp   -o int_ring
 * First do: module load openmpi/gcc/3.1.4
 */
#include <stdio.h>
#include <cstdlib>
#include <mpi.h>
#include <iostream>

double ring(long N_loops, int* msg, int msg_size, MPI_Comm comm) {
  int rank, p;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  int proc_next;
  int proc_current = 0;
  proc_next = proc_current + 1;
  
  MPI_Barrier(comm);
  double tt = MPI_Wtime();
  for (long repeat  = 0; repeat < N_loops; repeat++) {
    MPI_Status status;
    // Only rank 0 will print this
    //if (!rank) printf("Loop %d/%d\n", repeat, N_loops);
    for (int r=0; r<p; r++){
      proc_current = r;
      proc_next = (proc_current + 1) % p;
      if (rank == proc_current) {
        // This is the single integer case, so add to it.
        // Otherwise for the array case, don't need to modify.
        if (msg_size==1) {
          *msg += rank;
        }
        //printf("Loop %d: Proc %d just added to msg, now msg=%d. Sending to proc %d\n", repeat, rank, *msg, proc_next);
        MPI_Send(msg, msg_size, MPI_INT, proc_next, repeat, comm);
      }
      else if (rank == proc_next) {
        MPI_Recv(msg, msg_size, MPI_INT, proc_current, repeat, comm, &status);
        //printf("Loop %d: Proc %d just received msg (msg=%d)\n", repeat, rank, *msg);
      }
    }

  }
  tt = MPI_Wtime() - tt;

  return tt;
}


int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  if (argc < 3) {
    printf("Usage: mpirun ./int_ring <N_loops> <single_int>\n");
    abort();
  }
  long N_loops = atol(argv[1]);
  // Single_int should be 0 or 1; if 0, add to int
  // if 1, pass array
  int single_int = atoi(argv[2]);
  if (single_int!=0){
    if (single_int!=1){
      printf("single_int must be 0 or 1!\n");
      return -1;
    }
  }

  int rank, p;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  /* get name of host running MPI process */
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Rank %d/%d running on %s.\n", rank, p, processor_name);

  double tt;
  // Pass single integer
  if (single_int) {
    printf("Passing integer around network\n");
    int msg = 0;
    int msg_size = 1;
    tt = ring(N_loops, &msg, msg_size, comm);
    MPI_Finalize();

    if (!rank) printf("final message: %d\n", msg);
    /* Compute expected message */
    int msg_true = 0;
    for (int i=0; i<p; i++){
      msg_true += i;
    }
    msg_true *= N_loops;
    if (!rank) printf("expected message: %d\n", msg_true);
    if (!rank) printf("ring latency: %e ms\n", tt/(N_loops*p) * 1000);
  }
  // OR: Pass array
  else {
    // int = 4 bytes; 5e5 ints = 2MB
    long msg_size = 500000;
    printf("Passing array of %d ints (%f MB) around network loop\n", msg_size, msg_size*sizeof(int)/1e6);
    int* msg_arr = (int*) malloc(msg_size*sizeof(int));
    for (long i = 0; i < msg_size; i++) msg_arr[i] = 42;
    tt = ring(N_loops, msg_arr, msg_size, comm);
    MPI_Finalize();
    free(msg_arr);
    if (!rank) printf("ring latency: %e ms\n", tt/(N_loops*p) * 1000);
    if (!rank) printf("ring bandwidth: %e GB/s\n", (sizeof(int)*msg_size*N_loops*p)/tt/1e9);
  }
  return 0;
}

