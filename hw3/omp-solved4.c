/******************************************************************************
* FILE: omp_bug4.c
* DESCRIPTION:
*   This very simple program causes a segmentation fault.
* AUTHOR: Blaise Barney  01/09/04
* LAST REVISED: 04/06/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 1048

int main (int argc, char *argv[]) 
{
int nthreads, tid, i, j;
// Change a to a pointer to an int array.
//double a[N][N];
int *a;

/* Fork a team of threads with explicit variable scoping */
#pragma omp parallel shared(nthreads) private(i,j,tid,a)
  {
  // BUG/FIX: There is not enough stack space for each 
  // thread to have its own copy of a on the stack. We 
  // thus need to malloc a. We also convert a to an int
  // so that it takes up less memory, as it will only be 
  // filled with ints anyway (sum of i, j, tid, all ints).
  a = (int*) malloc(N*N*sizeof(int));

  /* Obtain/print thread info */
  tid = omp_get_thread_num();
  if (tid == 0) 
    {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d starting...\n", tid);

  /* Each thread works on its own private copy of the array */
  for (i=0; i<N; i++)
    for (j=0; j<N; j++)
      //a[i][j] = tid + i + j;
      // We need to properly access the values of a for
      // a malloced array
      a[i*N+j] = tid + i + j;

  /* For confirmation */
  // Fixed this printline to print out an integer, and access
  // the last value of a appropriately
  printf("Thread %d done. Last element= %d\n",tid,a[N*N-1]);

  }  /* All threads join master thread and disband */

  // We now have to free a
  free(a);
}

