#include <math.h>
#include <stdio.h>
#include "utils.h"
#include "intrin-wrapper.h"

// Headers for intrinsics
#ifdef __SSE__
#include <xmmintrin.h>
#endif
#ifdef __SSE2__
#include <emmintrin.h>
#endif
#ifdef __AVX__
#include <immintrin.h>
#endif


// coefficients in the Taylor series expansion of sin(x)
static constexpr double c3  = -1/(((double)2)*3);
static constexpr double c5  =  1/(((double)2)*3*4*5);
static constexpr double c7  = -1/(((double)2)*3*4*5*6*7);
static constexpr double c9  =  1/(((double)2)*3*4*5*6*7*8*9);
static constexpr double c11 = -1/(((double)2)*3*4*5*6*7*8*9*10*11);
// sin(x) = x + c3*x^3 + c5*x^5 + c7*x^7 + x9*x^9 + c11*x^11

static constexpr double c2  = -1/(((double)2));
static constexpr double c4  =  1/(((double)2)*3*4);
static constexpr double c6  = -1/(((double)2)*3*4*5*6);
static constexpr double c8  =  1/(((double)2)*3*4*5*6*7*8);
static constexpr double c10 = -1/(((double)2)*3*4*5*6*7*8*9*10);
// sin(x+pi/2) = cosx = 1 + c2*x^2 + c4*x^4 + c6*x^6 + x8*x^8 + c10*x^10

// Computing 4 at a time, because for the vectorized 
// cases we should be able to do 4 simultaneously.
void sin4_reference(double* sinx, const double* x) {
  for (long i = 0; i < 4; i++) sinx[i] = sin(x[i]);
}

void sin4_taylor(double* sinx, const double* x) {
  for (int i = 0; i < 4; i++) {
    double x1  = x[i];
    double x2  = x1 * x1;
    double x3  = x1 * x2;
    double x5  = x3 * x2;
    double x7  = x5 * x2;
    double x9  = x7 * x2;
    double x11 = x9 * x2;

    double s = x1;
    s += x3  * c3;
    s += x5  * c5;
    s += x7  * c7;
    s += x9  * c9;
    s += x11 * c11;
    sinx[i] = s;
  }
}


void sin4_taylor_extended(double* sinx, const double* x) {
  // To compute sinx at any x value, we check the x location
  // and choose whether to compute the sin or cos, based on
  // Euler's formula.
  // [This takes a long time! There must be a better way 
  // but I can't think of it right now.]
  for (int i = 0; i < 4; i++) {
    // n is how many pi/2 shifts we need to get to the small x regime
    int n = floor((x[i]+M_PI/4)/(M_PI/2));
    // xm is x shifted to the small x regime
    double xm = x[i] - n*M_PI/2;
    // If n is odd (x in [pi/4,3pi/4], [5pi/4, 7pi/4]...) use cosine
    if (abs(n)%2==1) { 
        double x1  = xm;
        double x2  = x1 * x1;
        double x4  = x2 * x2;
        double x6  = x4 * x2;
        double x8  = x6 * x2;
        double x10 = x8 * x2;

        double s = 1;
        s += x2  * c2;
        s += x4  * c4;
        s += x6  * c6;
        s += x8  * c8;
        s += x10 * c10;
        sinx[i] = s;
    }
    // If n is even (x in [-pi/4,pi/4], [3pi/4, 5pi/4]...) use sin
    else {
        double x1  = xm;
        double x2  = x1 * x1;
        double x3  = x1 * x2;
        double x5  = x3 * x2;
        double x7  = x5 * x2;
        double x9  = x7 * x2;
        double x11 = x9 * x2;

        double s = x1;
        s += x3  * c3;
        s += x5  * c5;
        s += x7  * c7;
        s += x9  * c9;
        s += x11 * c11;
        sinx[i] = s;
    }
    // If n is a multiple of 2 or (x in [3pi/4, 5pi/4] 
    // or [5pi/4, 7pi/4]) use the negative
    int nmod = (n % 4 + 4) % 4;
    if (nmod==2 || nmod==3){
        sinx[i] *= -1;
    }
  }
}

void sin4_intrin_extended(double* sinx, const double* x) {
  // Here I implement a method for sinx that can take any x,
  // not just in the range -pi/4, pi/4. This relies on a trick
  // using Euler's formula where we shift the angle and compute
  // cosx instead. Because of the vectorization, we here compute
  // both sinx and cosx for all for values of x, and at the end
  // figure out which we need.
  // [This is super inefficient and takes a long time! But I 
  // can't think of a better way right now.]
#if defined(__AVX__)
  double sinres[4];
  double cosres[4];
  double xm[4];
  int n[4];
  // Shift angle towards small x, keep track of how much shifting
  // we did (n)
  for (int i = 0; i < 4; i++) {
    n[i] = floor((x[i]+M_PI/4)/(M_PI/2));
    xm[i] = x[i] - n[i]*M_PI/2;
  }
  // sin
  __m256d x1, x2, x3, x5, x7, x9, x11;
  x1  = _mm256_load_pd(xm);
  x2  = _mm256_mul_pd(x1, x1);
  x3  = _mm256_mul_pd(x1, x2);
  x5  = _mm256_mul_pd(x3, x2);
  x7  = _mm256_mul_pd(x5, x2);
  x9  = _mm256_mul_pd(x7, x2);
  x11  = _mm256_mul_pd(x9, x2);

  __m256d s = x1;
  s = _mm256_add_pd(s, _mm256_mul_pd(x3 , _mm256_set1_pd(c3 )));
  s = _mm256_add_pd(s, _mm256_mul_pd(x5 , _mm256_set1_pd(c5 )));
  s = _mm256_add_pd(s, _mm256_mul_pd(x7 , _mm256_set1_pd(c7 )));
  s = _mm256_add_pd(s, _mm256_mul_pd(x9 , _mm256_set1_pd(c9 )));
  s = _mm256_add_pd(s, _mm256_mul_pd(x11 , _mm256_set1_pd(c11 )));
  _mm256_store_pd(sinres, s);

  // cos
  __m256d x4, x6, x8, x10;
  x4  = _mm256_mul_pd(x2, x2);
  x6  = _mm256_mul_pd(x4, x2);
  x8  = _mm256_mul_pd(x6, x2);
  x10  = _mm256_mul_pd(x8, x2);

  __m256d c = _mm256_set_pd(1.0, 1.0, 1.0, 1.0);
  c = _mm256_add_pd(c, _mm256_mul_pd(x2 , _mm256_set1_pd(c2 )));
  c = _mm256_add_pd(c, _mm256_mul_pd(x4 , _mm256_set1_pd(c4 )));
  c = _mm256_add_pd(c, _mm256_mul_pd(x6 , _mm256_set1_pd(c6 )));
  c = _mm256_add_pd(c, _mm256_mul_pd(x8 , _mm256_set1_pd(c8 )));
  c = _mm256_add_pd(c, _mm256_mul_pd(x10 , _mm256_set1_pd(c10 )));
  _mm256_store_pd(cosres, c);

  // Depending on the number of shifts we did, store the positive
  // or negative sin or cosine
  for (int i = 0; i < 4; i++) {
    int nmod = (n[i] % 4 + 4) % 4;
    if (nmod==0){ 
      sinx[i] = sinres[i]; 
    }
    else if (nmod==1){ 
      sinx[i] = cosres[i]; 
    }
    else if (nmod==2){ 
      sinx[i] = -sinres[i]; 
    }
    // nmod==3
    else { 
      sinx[i] = -cosres[i]; 
    }
  }
#else
  sin4_reference(sinx, x);
#endif
}


void sin4_intrin(double* sinx, const double* x) {
  // The definition of intrinsic functions can be found at:
  // https://software.intel.com/sites/landingpage/IntrinsicsGuide/#
#if defined(__AVX__)
  // I have extended the Taylor series evaluation of sinx 
  // to about 12 digit accuarcy, with AVX intrinsics.
  __m256d x1, x2, x3, x5, x7, x9, x11;
  x1  = _mm256_load_pd(x);
  x2  = _mm256_mul_pd(x1, x1);
  x3  = _mm256_mul_pd(x1, x2);
  x5  = _mm256_mul_pd(x3, x2);
  x7  = _mm256_mul_pd(x5, x2);
  x9  = _mm256_mul_pd(x7, x2);
  x11  = _mm256_mul_pd(x9, x2);

  __m256d s = x1;
  s = _mm256_add_pd(s, _mm256_mul_pd(x3 , _mm256_set1_pd(c3 )));
  s = _mm256_add_pd(s, _mm256_mul_pd(x5 , _mm256_set1_pd(c5 )));
  s = _mm256_add_pd(s, _mm256_mul_pd(x7 , _mm256_set1_pd(c7 )));
  s = _mm256_add_pd(s, _mm256_mul_pd(x9 , _mm256_set1_pd(c9 )));
  s = _mm256_add_pd(s, _mm256_mul_pd(x11 , _mm256_set1_pd(c11 )));
  _mm256_store_pd(sinx, s);
#elif defined(__SSE2__)
  constexpr int sse_length = 2;
  for (int i = 0; i < 4; i+=sse_length) {
    __m128d x1, x2, x3;
    x1  = _mm_load_pd(x+i);
    x2  = _mm_mul_pd(x1, x1);
    x3  = _mm_mul_pd(x1, x2);

    __m128 s = x1;
    s = _mm_add_pd(s, _mm_mul_pd(x3 , _mm_set1_pd(c3 )));
    _mm_store_pd(sinx+i, s);
  }
#else
  sin4_reference(sinx, x);
#endif
}

void sin4_vector(double* sinx, const double* x) {
  // The Vec class is defined in the file intrin-wrapper.h
  typedef Vec<double,4> Vec4;
  Vec4 x1, x2, x3;
  x1  = Vec4::LoadAligned(x);
  x2  = x1 * x1;
  x3  = x1 * x2;

  Vec4 s = x1;
  s += x3  * c3 ;
  s.StoreAligned(sinx);
}

double err(double* x, double* y, long N) {
  double error = 0;
  for (long i = 0; i < N; i++) error = std::max(error, fabs(x[i]-y[i]));
  return error;
}

int main() {
  Timer tt;
  long N = 1000000;
  double* x = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_ref = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_taylor = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_intrin = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_vector = (double*) aligned_malloc(N*sizeof(double));

  double* sinx_taylor_extended = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_intrin_extended = (double*) aligned_malloc(N*sizeof(double));
  for (long i = 0; i < N; i++) {
    //x[i] = (drand48()-0.5) * M_PI/2; // [-pi/4,pi/4]
    x[i] = (drand48()-0.5) * 4*M_PI; // [-pi,pi]
    sinx_ref[i] = 0;
    sinx_taylor[i] = 0;
    sinx_intrin[i] = 0;
    sinx_vector[i] = 0;

    sinx_taylor_extended[i] = 0;
    sinx_intrin_extended[i] = 0;
  }

  // This is giving the memory address for index i of the length N
  // array, jumping by 4 to leave room for the 4 solutions for the 4 xs.
  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_reference(sinx_ref+i, x+i);
    }
  }
  printf("Reference time: %6.4f\n", tt.toc());

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_taylor(sinx_taylor+i, x+i);
    }
  }
  printf("Taylor time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_ref, sinx_taylor, N));

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_intrin(sinx_intrin+i, x+i);
    }
  }
  printf("Intrin time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_ref, sinx_intrin, N));

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_vector(sinx_vector+i, x+i);
    }
  }
  printf("Vector time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_ref, sinx_vector, N));

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_taylor_extended(sinx_taylor_extended+i, x+i);
    }
  }
  printf("Taylor extended time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_ref, sinx_taylor_extended, N));

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_intrin_extended(sinx_intrin_extended+i, x+i);
    }
  }
  printf("Intrin extended time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_ref, sinx_intrin_extended, N));

  aligned_free(x);
  aligned_free(sinx_ref);
  aligned_free(sinx_taylor);
  aligned_free(sinx_taylor_extended);
  aligned_free(sinx_intrin);
  aligned_free(sinx_intrin_extended);
  aligned_free(sinx_vector);
}

