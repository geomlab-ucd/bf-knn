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

#ifndef BF_KNN_DEVICE_CUH_
#define BF_KNN_DEVICE_CUH_

#define TILE_SIZE 96
#define CHUNK_SIZE 16
#define BLOCK_SIZE 16

__global__ void kComputeDistances(const int K, const int M, const int N,
                                  const float* A, const float* B, long long* C);

// Knn_NV >= nn
#define K500_NT 64
#define K500_VT 9
#define K500_NV K500_NT * K500_VT
#define K1000_NT 128
#define K1000_VT 11
#define K1000_NV K1000_NT * K1000_VT
#define K2000_NT 256
#define K2000_VT 11
#define K2000_NV K2000_NT * K2000_VT
#define K3000_NT 512
#define K3000_VT 6
#define K3000_NV K3000_NT * K3000_VT

template <int NT, int VT, int NV>
__global__ void kSortCandidateGroups(const int nr, const int pnr, long long* C);

template <int NT, int VT, int NV>
__global__ void kMergeCandidateGroups(const int nr, const int pnr,
                                      const int span, long long* C);

__global__ void kRetrieveResults(const int pnr, const int nn,
                                 const long long* const C, int* const idx,
                                 float* const dist);

#endif  // BF_KNN_DEVICE_CUH_
