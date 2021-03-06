/******************************************************************************
* FILE: omp_bug2.c
* DESCRIPTION:
*   Another OpenMP program with a bug. 
* AUTHOR: Blaise Barney 
* LAST REVISED: 04/06/05 
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[]) 
{
int nthreads, i, tid;
// BUG/FIX: A float didn't give enough precision for
// this problem, so we changed it to a double.
double total;

/*** Spawn parallel region ***/
// BUG/FIX: We had to make tid private, so that each
// thread can separately have its own tid.
#pragma omp parallel private(tid)
  {
  /* Obtain thread number */
  tid = omp_get_thread_num();
  /* Only master thread does this */
  if (tid == 0) {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d is starting...\n",tid);

  #pragma omp barrier

  /* do some work */
  total = 0.0;
  // BUG/FIX: We needed a reduction here so that all threads
  // can add to the variable "total" safely.
  #pragma omp for reduction(+: total) schedule(dynamic,10)
  for (i=0; i<1000000; i++) 
     total = total + i*1.0;

  printf ("Thread %d is done! Total= %e\n",tid,total);

  } /*** End of parallel region ***/
}
