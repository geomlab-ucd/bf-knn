/*******************************************************************************
 * bf-knn (Brute-Force k-Nearest Neighbors Search on the GPU) is the proprietary
 * property of The Regents of the University of California ("The Regents.")
 *
 * Copyright © 2015 The Regents of the University of California, Davis campus.
 * All Rights Reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted by nonprofit, research institutions for research
 * use only, provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * The name of The Regents may not be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * The end-user understands that the program was developed for research purposes
 * and is advised not to rely exclusively on the program for any reason.
 *
 * THE SOFTWARE PROVIDED IS ON AN "AS IS" BASIS, AND THE REGENTS HAVE NO
 * OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
 * MODIFICATIONS. THE REGENTS SPECIFICALLY DISCLAIM ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
 * EVENT SHALL THE REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
 * INCIDENTAL, EXEMPLARY OR CONSEQUENTIAL DAMAGES, INCLUDING BUT NOT LIMITED TO
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES, LOSS OF USE, DATA OR PROFITS, OR
 * BUSINESS INTERRUPTION, HOWEVER CAUSED AND UNDER ANY THEORY OF LIABILITY
 * WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE AND ITS
 * DOCUMENTATION, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * If you do not agree to these terms, do not download or use the software. This
 * license may be modified only in a writing signed by authorized signatory of
 * both parties.
 *
 * For commercial license information please contact copyright@ucdavis.edu.
 ******************************************************************************/

#ifndef UTIL_H_
#define UTIL_H_

#include <cassert>
#include <cstdio>

#define CHECK_ERROR(call)                                              \
  do {                                                                 \
    cudaError_t err = call;                                            \
    if (err != cudaSuccess) {                                          \
      fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

#define SYNC_AND_CHECK_ERROR                                           \
  do {                                                                 \
    cudaDeviceSynchronize();                                           \
    cudaError_t err = cudaGetLastError();                              \
    if (err != cudaSuccess) {                                          \
      fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

class GpuTimer {
 public:
  GpuTimer() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }

  ~GpuTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void Start() { cudaEventRecord(start_); }

  void Stop() {
    cudaEventRecord(stop_);
    cudaEventSynchronize(stop_);
  }

  void CalculateElapsed() { cudaEventElapsedTime(&elapsed_, start_, stop_); }

  float elapsed() const { return elapsed_; }

 private:
  cudaEvent_t start_;
  cudaEvent_t stop_;
  float elapsed_;
};

#define CREATE_AND_START_TIMER \
  GpuTimer gpu_timer;          \
  gpu_timer.Start();

#define STOP_TIMER_AND_CALCULATE_ELAPSED \
  gpu_timer.Stop();                      \
  gpu_timer.CalculateElapsed();

inline int DivideAndCeil(int x, int f) {
  assert(x >= 0);
  assert(f > 0);
  return (x + f - 1) / f;
}

inline int DivideAndFloor(int x, int f) {
  assert(x >= 0);
  assert(f > 0);
  return x / f;
}

inline int CeilToMultiple(int x, int f) {
  assert(x >= 0);
  assert(f > 0);
  return (x + f - 1) / f * f;
}

#endif  // UTIL_H_
