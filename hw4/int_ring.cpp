/* Run with: mpicxx  -std=c++11 -O3 -march=native -fopenmp    int_ring.cpp   -o int_ring
 * First do: module load openmpi/gcc/3.1.4
 */
#include <stdio.h>
#include <cstdlib>
#include <mpi.h>
#include <iostream>

double ring(long N_loops, int* msg, MPI_Comm comm) {
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
    if (!rank) printf("Loop %d/%d\n", repeat, N_loops);
    for (int r=0; r<p; r++){
      proc_current = r;
      proc_next = (proc_current + 1) % p;
      if (rank == proc_current) {
        *msg += rank;
        printf("Loop %d: Proc %d just added to msg, now msg=%d. Sending to proc %d\n", repeat, rank, *msg, proc_next);
        MPI_Send(msg, 1, MPI_INT, proc_next, repeat, comm);
      }
      else if (rank == proc_next) {
        MPI_Recv(msg, 1, MPI_INT, proc_current, repeat, comm, &status);
        printf("Loop %d: Proc %d just received msg (msg=%d)\n", repeat, rank, *msg);
      }
    }
    //if (rank == proc_current) {
    //  // add own rank to message
    //  msg += rank;
    //  printf("Proc %d just added to msg, now msg=%d. Sending to proc %d\n", rank, msg, proc_next);
    //  // 1 is size; single integer so just 1
    //  // repeat is the message tag, should be unique
    //  MPI_Send(&msg, 1, MPI_INT, proc_next, repeat, comm);
    //}
    //else if (rank == proc_next) {
    //  MPI_Recv(&msg, 1, MPI_INT, proc_current, repeat, comm, &status);
    //  printf("Proc %d just received msg (msg=%d)\n", rank, msg);
    //  proc_current = proc_next;
    //  proc_next = proc_current + 1;
    //}

  }
  tt = MPI_Wtime() - tt;

  return tt;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  if (argc < 2) {
    printf("Usage: mpirun ./int_ring <N_loops>\n");
    abort();
  }
  long N_loops = atol(argv[1]);

  int rank, p;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  /* get name of host running MPI process */
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Rank %d/%d running on %s.\n", rank, p, processor_name);

  int msg = 0;
  double tt = ring(N_loops, &msg, comm);
  MPI_Finalize();
  if (!rank) printf("final message: %d\n", msg);

  /* Compute expected message */
  int msg_true = 0;
  for (int i=0; i<p; i++){
    msg_true += i;
  }
  msg_true *= N_loops;
  if (!rank) printf("expected message: %d\n", msg_true);
  
  if (!rank) printf("ring latency: %e ms\n", tt/N_loops * 1000);

  //Nrepeat = 10000;
  //long Nsize = 1000000;
  //tt = time_pingpong(proc0, proc1, Nrepeat, Nsize, comm);
  //if (!rank) printf("pingpong bandwidth: %e GB/s\n", (Nsize*Nrepeat)/tt/1e9);

}

