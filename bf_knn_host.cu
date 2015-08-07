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

#include "bf_knn_host.h"

#include "util.h"
#include "bf_knn_device.cuh"

const bool kPrintTime = true;

void ComputeDistances(const int padded_num_dimension,
                      const int padded_num_query,
                      const int padded_num_reference,
                      const float* const d_padded_query,
                      const float* const d_padded_reference,
                      long long* const d_candidate) {
  assert(padded_num_dimension % CHUNK_SIZE == 0);
  assert(padded_num_query % TILE_SIZE == 0);
  assert(padded_num_reference % TILE_SIZE == 0);

  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(padded_num_reference / TILE_SIZE, padded_num_query / TILE_SIZE);

  CREATE_AND_START_TIMER;

  kComputeDistances<<<grid, block>>>
      (padded_num_dimension, padded_num_query, padded_num_reference,
       d_padded_query, d_padded_reference, d_candidate);

  SYNC_AND_CHECK_ERROR;

  STOP_TIMER_AND_CALCULATE_ELAPSED;

  if (kPrintTime) printf("ComputeDistances %.3f ms\n", gpu_timer.elapsed());
}

void SortCandidateGroups(const int num_query, const int num_reference,
                         const int padded_num_reference,
                         const int num_nearest_neighbor,
                         long long* const d_candidate) {
  dim3 block, grid;
  if (num_nearest_neighbor == 500) {
    block = dim3(K500_NT);
    grid = dim3(DivideAndCeil(num_reference, K500_NV), num_query);
  } else if (num_nearest_neighbor == 1000) {
    block = dim3(K1000_NT);
    grid = dim3(DivideAndCeil(num_reference, K1000_NV), num_query);
  } else if (num_nearest_neighbor == 2000) {
    block = dim3(K2000_NT);
    grid = dim3(DivideAndCeil(num_reference, K2000_NV), num_query);
  } else if (num_nearest_neighbor == 3000) {
    block = dim3(K3000_NT);
    grid = dim3(DivideAndCeil(num_reference, K3000_NV), num_query);
  }

  CREATE_AND_START_TIMER;

  if (num_nearest_neighbor == 500) {
    kSortCandidateGroups<K500_NT, K500_VT, K500_NV><<<grid, block>>>
        (num_reference, padded_num_reference, d_candidate);
  } else if (num_nearest_neighbor == 1000) {
    kSortCandidateGroups<K1000_NT, K1000_VT, K1000_NV><<<grid, block>>>
        (num_reference, padded_num_reference, d_candidate);
  } else if (num_nearest_neighbor == 2000) {
    kSortCandidateGroups<K2000_NT, K2000_VT, K2000_NV><<<grid, block>>>
        (num_reference, padded_num_reference, d_candidate);
  } else if (num_nearest_neighbor == 3000) {
    kSortCandidateGroups<K3000_NT, K3000_VT, K3000_NV><<<grid, block>>>
        (num_reference, padded_num_reference, d_candidate);
  }

  SYNC_AND_CHECK_ERROR;

  STOP_TIMER_AND_CALCULATE_ELAPSED;

  if (kPrintTime) printf("SortCandidateGroups %.3f ms\n", gpu_timer.elapsed());
}

void MergeCandidateGroups(const int num_query, const int num_reference,
                          const int padded_num_reference,
                          const int num_nearest_neighbor,
                          long long* const d_candidate) {
  dim3 block;
  int remaining;
  if (num_nearest_neighbor == 500) {
    block = dim3(K500_NT);
    remaining = DivideAndCeil(num_reference, K500_NV);
  } else if (num_nearest_neighbor == 1000) {
    block = dim3(K1000_NT);
    remaining = DivideAndCeil(num_reference, K1000_NV);
  } else if (num_nearest_neighbor == 2000) {
    block = dim3(K2000_NT);
    remaining = DivideAndCeil(num_reference, K2000_NV);
  } else if (num_nearest_neighbor == 3000) {
    block = dim3(K3000_NT);
    remaining = DivideAndCeil(num_reference, K3000_NV);
  }

  float total_elapsed = 0.0f;

  while (remaining > 1) {
    int batch = DivideAndFloor(remaining, 2);
    int span = DivideAndCeil(remaining, 2);

    dim3 grid(batch, num_query);

    CREATE_AND_START_TIMER;

    if (num_nearest_neighbor == 500) {
      kMergeCandidateGroups<K500_NT, K500_VT, K500_NV><<<grid, block>>>
          (num_reference, padded_num_reference, span, d_candidate);
    } else if (num_nearest_neighbor == 1000) {
      kMergeCandidateGroups<K1000_NT, K1000_VT, K1000_NV><<<grid, block>>>
          (num_reference, padded_num_reference, span, d_candidate);
    } else if (num_nearest_neighbor == 2000) {
      kMergeCandidateGroups<K2000_NT, K2000_VT, K2000_NV><<<grid, block>>>
          (num_reference, padded_num_reference, span, d_candidate);
    } else if (num_nearest_neighbor == 3000) {
      kMergeCandidateGroups<K3000_NT, K3000_VT, K3000_NV><<<grid, block>>>
          (num_reference, padded_num_reference, span, d_candidate);
    }

    SYNC_AND_CHECK_ERROR;

    STOP_TIMER_AND_CALCULATE_ELAPSED;

    if (kPrintTime) total_elapsed += gpu_timer.elapsed();

    remaining = span;
  }

  if (kPrintTime) printf("MergeCandidateGroups %.3f ms\n", total_elapsed);
}

void RetrieveResults(const int num_query, const int padded_num_reference,
                     const int num_nearest_neighbor,
                     const long long* const d_candidate, int* const d_knn_index,
                     float* const d_knn_distance) {
  dim3 block(min(num_nearest_neighbor, 1024));
  dim3 grid(DivideAndCeil(num_nearest_neighbor, 1024), num_query);

  CREATE_AND_START_TIMER;

  kRetrieveResults<<<grid, block>>>
      (padded_num_reference, num_nearest_neighbor, d_candidate, d_knn_index,
       d_knn_distance);

  SYNC_AND_CHECK_ERROR;

  STOP_TIMER_AND_CALCULATE_ELAPSED;

  if (kPrintTime) printf("RetrieveResults %.3f ms\n", gpu_timer.elapsed());
}

void BruteForceKnnSearch(const int num_dimension, const int num_query,
                         const int num_reference,
                         const int num_nearest_neighbor,
                         const float* const query, const float* const reference,
                         int* const knn_index, float* const knn_distance) {
  assert(num_dimension > 0);
  assert(num_query > 0);
  assert(num_reference > 0);
  assert(num_nearest_neighbor == 500 || num_nearest_neighbor == 1000 ||
         num_nearest_neighbor == 2000 || num_nearest_neighbor == 3000);
  assert(num_reference >= num_nearest_neighbor);
  assert(query != NULL);
  assert(reference != NULL);
  assert(knn_index != NULL);
  assert(knn_distance != NULL);

  // The reason that 'query' and 'reference' are padded is kComputeDistances
  // only works for a complete 96 X 96 tile of 'candidate' and processes a chunk
  // of 16 dimensions in each iteration.

  const int padded_num_query = CeilToMultiple(num_query, TILE_SIZE);
  const int padded_num_reference = CeilToMultiple(num_reference, TILE_SIZE);
  const int padded_num_dimension = CeilToMultiple(num_dimension, CHUNK_SIZE);

  float* h_padded_query = new float[padded_num_dimension * padded_num_query];
  float* h_padded_reference =
      new float[padded_num_dimension * padded_num_reference];

  float* d_padded_query = NULL;
  CHECK_ERROR(
      cudaMalloc((void**)&d_padded_query,
                 sizeof(float) * padded_num_dimension * padded_num_query));
  float* d_padded_reference = NULL;
  CHECK_ERROR(
      cudaMalloc((void**)&d_padded_reference,
                 sizeof(float) * padded_num_dimension * padded_num_reference));
  long long* d_candidate = NULL;
  CHECK_ERROR(
      cudaMalloc((void**)&d_candidate,
                 sizeof(long long) * padded_num_query * padded_num_reference));
  int* d_knn_index = NULL;
  CHECK_ERROR(cudaMalloc((void**)&d_knn_index,
                         sizeof(int) * num_query * num_nearest_neighbor));
  float* d_knn_distance = NULL;
  CHECK_ERROR(cudaMalloc((void**)&d_knn_distance,
                         sizeof(float) * num_query * num_nearest_neighbor));

  memset((void*)h_padded_query, 0,
         sizeof(float) * padded_num_dimension * padded_num_query);
  for (int i = 0; i < num_dimension; ++i)
    memcpy(h_padded_query + padded_num_query * i, query + num_query * i,
           sizeof(float) * num_query);
  memset((void*)h_padded_reference, 0,
         sizeof(float) * padded_num_dimension * padded_num_reference);
  for (int i = 0; i < num_dimension; ++i)
    memcpy(h_padded_reference + padded_num_reference * i,
           reference + num_reference * i, sizeof(float) * num_reference);

  CHECK_ERROR(
      cudaMemcpy(d_padded_query, h_padded_query,
                 sizeof(float) * padded_num_dimension * padded_num_query,
                 cudaMemcpyHostToDevice));
  CHECK_ERROR(
      cudaMemcpy(d_padded_reference, h_padded_reference,
                 sizeof(float) * padded_num_dimension * padded_num_reference,
                 cudaMemcpyHostToDevice));

  ComputeDistances(padded_num_dimension, padded_num_query, padded_num_reference,
                   d_padded_query, d_padded_reference, d_candidate);

  SortCandidateGroups(num_query, num_reference, padded_num_reference,
                      num_nearest_neighbor, d_candidate);

  MergeCandidateGroups(num_query, num_reference, padded_num_reference,
                       num_nearest_neighbor, d_candidate);

  RetrieveResults(num_query, padded_num_reference, num_nearest_neighbor,
                  d_candidate, d_knn_index, d_knn_distance);

  CHECK_ERROR(cudaMemcpy(knn_index, d_knn_index,
                         sizeof(int) * num_query * num_nearest_neighbor,
                         cudaMemcpyDeviceToHost));
  CHECK_ERROR(cudaMemcpy(knn_distance, d_knn_distance,
                         sizeof(float) * num_query * num_nearest_neighbor,
                         cudaMemcpyDeviceToHost));

  delete[] h_padded_query;
  delete[] h_padded_reference;

  CHECK_ERROR(cudaFree(d_padded_query));
  CHECK_ERROR(cudaFree(d_padded_reference));
  CHECK_ERROR(cudaFree(d_candidate));
  CHECK_ERROR(cudaFree(d_knn_index));
  CHECK_ERROR(cudaFree(d_knn_distance));
}
