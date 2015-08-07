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

#include "bf_knn_device.cuh"

#include "moderngpu.cuh"

__global__ void kComputeDistances(const int K, const int M, const int N,
                                  const float* A, const float* B,
                                  long long* C) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tid = ty * 16 + tx;
  int tx2 = tid % 32;
  int ty2 = tid / 32;

  volatile __shared__ float as[16][96];
  volatile __shared__ float bs[16][96];

  float cr[6][6];
  float ar;
  float br[6];

  float asr[2][3];
  float bsr[2][3];

  A += ty2 * M + (by * 96 + tx2);
  B += ty2 * N + (bx * 96 + tx2);
  C += (by * 96 + ty) * N + (bx * 96 + tx);

  // Zero C reg
  #pragma unroll
  for (int i = 0; i < 6; ++i)
    #pragma unroll
    for (int j = 0; j < 6; ++j) cr[i][j] = 0.0f;

  // Load A gmem->smem
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    #pragma unroll
    for (int j = 0; j < 3; ++j) as[i * 8 + ty2][j * 32 + tx2] = A[j * 32];
    A += M * 8;
  }

  // Load B gmem->smem
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    #pragma unroll
    for (int j = 0; j < 3; ++j) bs[i * 8 + ty2][j * 32 + tx2] = B[j * 32];
    B += N * 8;
  }

  __syncthreads();

  for (int kk = 0; kk < K - 16; kk += 16) {
    // Load A gmen->reg
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
      #pragma unroll
      for (int j = 0; j < 3; ++j) asr[i][j] = A[j * 32];
      A += M * 8;
    }

    // Load B gmem->reg
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
      #pragma unroll
      for (int j = 0; j < 3; ++j) bsr[i][j] = B[j * 32];
      B += N * 8;
    }

    // Compute
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
      // Load B smen->reg
      #pragma unroll
      for (int j = 0; j < 6; ++j) br[j] = bs[k][j * 16 + tx];

      #pragma unroll
      for (int i = 0; i < 6; ++i) {
        ar = as[k][i * 16 + ty];
        #pragma unroll
        for (int j = 0; j < 6; ++j) {
          float d = ar - br[j];
          cr[i][j] += d * d;
        }
      }
    }

    __syncthreads();

    // Load A reg->smem
    #pragma unroll
    for (int i = 0; i < 2; ++i)
      #pragma unroll
      for (int j = 0; j < 3; ++j) as[i * 8 + ty2][j * 32 + tx2] = asr[i][j];

    // Load B reg->smem
    #pragma unroll
    for (int i = 0; i < 2; ++i)
      #pragma unroll
      for (int j = 0; j < 3; ++j) bs[i * 8 + ty2][j * 32 + tx2] = bsr[i][j];

    __syncthreads();
  }

  // Compute last 16 dimensions
  #pragma unroll
  for (int k = 0; k < 16; ++k) {
    // Load B smen->reg
    #pragma unroll
    for (int j = 0; j < 6; ++j) br[j] = bs[k][j * 16 + tx];

    #pragma unroll
    for (int i = 0; i < 6; ++i) {
      ar = as[k][i * 16 + ty];
      #pragma unroll
      for (int j = 0; j < 6; ++j) {
        float d = ar - br[j];
        cr[i][j] += d * d;
      }
    }
  }

  // Store C reg->gmem
  #pragma unroll
  for (int i = 0; i < 6; ++i) {
    #pragma unroll
    for (int j = 0; j < 6; ++j) {
      long long c = (long long)__float_as_int(cr[i][j]);
      c = (c << 32) | (bx * 96 + j * 16 + tx);
      C[j * 16] = c;
    }
    C += N * 16;
  }
}

template <int NT, int VT, int NV>
__global__ void kSortCandidateGroups(const int nr, const int pnr,
                                     long long* C) {
  int tid = threadIdx.x;

  __shared__ long long cs[NT * (VT + 1)];

  int offset = blockIdx.x * NV;
  C += blockIdx.y * pnr + offset;

  long long cr[VT];
  int ir[VT];

  #pragma unroll
  for (int i = 0; i < VT; ++i) {
    int index = NT * i + tid;
    if (offset + index < nr)
      cr[i] = C[index];
    else
      cr[i] = LLONG_MAX;
  }
  __syncthreads();

  mgpu::CTAMergesort<NT, VT, false, false>(cr, ir, cs, (int*)0, NV, tid,
                                           mgpu::less<long long>());

  #pragma unroll
  for (int i = 0; i < VT; ++i) {
    int index = NT * i + tid;
    if (offset + index < pnr) C[index] = cs[index];
  }
}

template __global__ void kSortCandidateGroups
    <K500_NT, K500_VT, K500_NV>(const int nr, const int pnr, long long* C);
template __global__ void kSortCandidateGroups
    <K1000_NT, K1000_VT, K1000_NV>(const int nr, const int pnr, long long* C);
template __global__ void kSortCandidateGroups
    <K2000_NT, K2000_VT, K2000_NV>(const int nr, const int pnr, long long* C);
template __global__ void kSortCandidateGroups
    <K3000_NT, K3000_VT, K3000_NV>(const int nr, const int pnr, long long* C);

template <int NT, int VT, int NV>
__global__ void kMergeCandidateGroups(const int nr, const int pnr,
                                      const int span, long long* C) {
  int tid = threadIdx.x;

  __shared__ long long cs[NV * 2];  // candidates in shared memory

  C += blockIdx.y * pnr + blockIdx.x * NV;
  long long* D = C + span * NV;
  int offset_d = blockIdx.x * NV + span * NV;

  #pragma unroll
  for (int i = 0; i < VT; ++i) {
    int index = NT * i + tid;
    cs[index] = C[index];
  }
  #pragma unroll
  for (int i = 0; i < VT; ++i) {
    int index = NT * i + tid;
    if (offset_d + index < nr)
      cs[NV + index] = D[index];
    else
      cs[NV + index] = LLONG_MAX;
  }
  __syncthreads();

  int diag = VT * tid;
  int mp = mgpu::MergePath<mgpu::MgpuBoundsLower>(cs, NV, cs + NV, NV, diag,
                                                  mgpu::less<long long>());

  long long cr[VT];  // candidates in registers
  int ir[VT];
  mgpu::SerialMerge<VT, true>(cs, mp, NV, NV + diag - mp, NV * 2, cr, ir,
                              mgpu::less<long long>());

  #pragma unroll
  for (int i = 0; i < VT; ++i) {
    C[VT * tid + i] = cr[i];
  }
}

template __global__ void kMergeCandidateGroups
    <K500_NT, K500_VT, K500_NV>(const int nr, const int pnr, const int span,
                                long long* C);
template __global__ void kMergeCandidateGroups
    <K1000_NT, K1000_VT, K1000_NV>(const int nr, const int pnr, const int span,
                                   long long* C);
template __global__ void kMergeCandidateGroups
    <K2000_NT, K2000_VT, K2000_NV>(const int nr, const int pnr, const int span,
                                   long long* C);
template __global__ void kMergeCandidateGroups
    <K3000_NT, K3000_VT, K3000_NV>(const int nr, const int pnr, const int span,
                                   long long* C);

__global__ void kRetrieveResults(const int pnr, const int nn,
                                 const long long* const C, int* const idx,
                                 float* const dist) {
  int qid = blockIdx.y;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= nn) return;

  long long c = C[qid * pnr + tid];
  idx[qid * nn + tid] = c & 0xFFFFFFFF;
  dist[qid * nn + tid] = sqrt(__int_as_float(c >> 32));
}
