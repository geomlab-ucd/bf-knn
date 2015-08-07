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

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "bf_knn_host.h"

int main(int argc, char** argv) {
  if (argc == 1) {
    printf("Required arguments:\n");
    printf("  -nd  Number of dimensions\n");
    printf("  -nq  Number of queries\n");
    printf("  -nr  Number of references\n");
    printf("  -nn  Number of nearest neighbors\n");
    printf("  -fq  Filename of queries\n");
    printf("  -fr  Filename of references\n");
    return 0;
  }

  // Example:
  // ./bf-knn -nd 50 -nq 500 -nr 100000 -nn 1000 -fq query.bin -fr reference.bin

  int num_dimension;
  int num_query;
  int num_reference;
  int num_nearest_neighbor;
  char filename_query[100];
  char filename_reference[100];

  int i = 1;
  while (i + 1 < argc) {  // Each option name is followed by an option value.
    if (strcmp(argv[i], "-nd") == 0) {
      num_dimension = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-nq") == 0) {
      num_query = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-nr") == 0) {
      num_reference = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-nn") == 0) {
      num_nearest_neighbor = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-fq") == 0) {
      strcpy(filename_query, argv[++i]);
    } else if (strcmp(argv[i], "-fr") == 0) {
      strcpy(filename_reference, argv[++i]);
    } else {
      printf("Unknown argument: %s\n", argv[i]);
      exit(EXIT_FAILURE);
    }
    ++i;
  }

  float* query = new float[num_dimension * num_query];
  float* reference = new float[num_dimension * num_reference];
  int* knn_index = new int[num_query * num_nearest_neighbor];
  float* knn_distance = new float[num_query * num_nearest_neighbor];

  // Query and reference points are represented by two row-major order 2D
  // matrices ('num_dimension' X 'num_query'/'num_reference') and stored as 1D
  // arrays of float values in two binary files. i.e., the components of the
  // same dimension are saved contiguously in file.

  FILE* file_query = fopen(filename_query, "rb");
  assert(fread(query, sizeof(float), num_dimension * num_query, file_query) ==
         num_dimension * num_query);
  fclose(file_query);

  FILE* file_reference = fopen(filename_reference, "rb");
  assert(fread(reference, sizeof(float), num_dimension * num_reference,
               file_reference) == num_dimension * num_reference);
  fclose(file_reference);

  BruteForceKnnSearch(num_dimension, num_query, num_reference,
                      num_nearest_neighbor, query, reference, knn_index,
                      knn_distance);

  // 'knn_index' and 'knn_distance' are row-major order 2D matrices
  // ('num_query' X 'num_nearest_neighbor') stored as 1D arrays. i.e., the
  // nearest neighbors of one particular query are saved contiguously in memory.

  FILE* file_knn_index = fopen("bf-knn_index.txt", "w");
  for (int i = 0; i < num_query; ++i) {
    for (int j = 0; j < num_nearest_neighbor; ++j) {
      if (j > 0) fprintf(file_knn_index, " ");
      fprintf(file_knn_index, "%d", knn_index[i * num_nearest_neighbor + j]);
    }
    fprintf(file_knn_index, "\n");
  }
  fclose(file_knn_index);

  FILE* file_knn_distance = fopen("bf-knn_distance.txt", "w");
  for (int i = 0; i < num_query; ++i) {
    for (int j = 0; j < num_nearest_neighbor; ++j) {
      if (j > 0) fprintf(file_knn_distance, " ");
      fprintf(file_knn_distance, "%.5f",
              knn_distance[i * num_nearest_neighbor + j]);
    }
    fprintf(file_knn_distance, "\n");
  }
  fclose(file_knn_distance);

  delete[] query;
  delete[] reference;
  delete[] knn_index;
  delete[] knn_distance;

  return 0;
}
