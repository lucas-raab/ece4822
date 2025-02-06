#include <stdio.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#define RANGE 2
#define DEBUG true

bool gensig(float* x, long N,bool isSin);

bool autocor(float* R, float* x, long N, long K);

bool autocor(float* R, float* x, long N, long K) {

  for (long k = 0; k < K; k++) {
      R[k] = 0;
      for (long n = 0; n < N - k; n++) {
	R[k] += x[n] * x[n + k];
      }

    }
    return true;
  }



#define PI 3.14159265358979323846
#define FREQUENCY .1       // Frequency of the sine wave in Hz
#define AMPLITUDE 1.0     // Amplitude of the sine wave

bool gensig(float* x, long N, bool isSin){

  if(isSin){
    for(int i = 0; i < N; i++)
      x[i] = AMPLITUDE * sin(2 * PI * FREQUENCY * i);

  } else{
    for(int i = 0; i < N; i++)
      x[i] = RANGE * ((float)rand() / RAND_MAX) - RANGE/2;
  }
  
  


  return true;
}
