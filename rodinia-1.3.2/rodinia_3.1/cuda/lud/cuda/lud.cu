/*
 * =====================================================================================
 *
 *       Filename:  lud.cu
 *
 *    Description:  The main wrapper for the suite
 *
 *        Version:  1.0
 *        Created:  10/22/2009 08:40:34 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Liang Wang (lw2aw), lw2aw@virginia.edu
 *        Company:  CS@UVa
 *
 * =====================================================================================
 */

#include <cuda.h>
#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <assert.h>

#include "common.h"

#ifdef RD_WG_SIZE_0_0
        #define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE RD_WG_SIZE
#else
        #define BLOCK_SIZE 16
#endif

static int do_verify = 0;

static struct option long_options[] = {
  /* name, has_arg, flag, val */
  {"input", 1, NULL, 'i'},
  {"size", 1, NULL, 's'},
  {"verify", 0, NULL, 'v'},
  {0,0,0,0}
};

extern void
lud_cuda(float *d_m, int matrix_dim);


int
main ( int argc, char *argv[] )
{
  printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);

  int matrix_dim = 32; /* default matrix_dim */
  int opt, option_index=0;
  func_ret_t ret;
  const char *input_file = NULL;
  float *m, *d_m, *mm;
  stopwatch sw;

  while ((opt = getopt_long(argc, argv, "::vs:i:", 
                            long_options, &option_index)) != -1 ) {
    switch(opt){
    case 'i':
      input_file = optarg;
      break;
    case 'v':
      do_verify = 1;
      break;
    case 's':
      matrix_dim = atoi(optarg);
      printf("Generate input matrix internally, size =%d\n", matrix_dim);
      // fprintf(stderr, "Currently not supported, use -i instead\n");
      // fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
      // exit(EXIT_FAILURE);
      break;
    case '?':
      fprintf(stderr, "invalid option\n");
      break;
    case ':':
      fprintf(stderr, "missing argument\n");
      break;
    default:
      fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n",
	      argv[0]);
      exit(EXIT_FAILURE);
    }
  }
  
  if ( (optind < argc) || (optind == 1)) {
    fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  if (input_file) {
    printf("Reading matrix from file %s\n", input_file);
    ret = create_matrix_from_file(&m, input_file, &matrix_dim);
    if (ret != RET_SUCCESS) {
      m = NULL;
      fprintf(stderr, "error create matrix from file %s\n", input_file);
      exit(EXIT_FAILURE);
    }
  } 
  else if (matrix_dim) {
    printf("Creating matrix internally size=%d\n", matrix_dim);
    ret = create_matrix(&m, matrix_dim);
    if (ret != RET_SUCCESS) {
      m = NULL;
      fprintf(stderr, "error create matrix internally size=%d\n", matrix_dim);
      exit(EXIT_FAILURE);
    }
  }


  else {
    printf("No input file specified!\n");
    exit(EXIT_FAILURE);
  }

  if (do_verify){
    printf("Before LUD\n");
    // print_matrix(m, matrix_dim);
    matrix_duplicate(m, &mm, matrix_dim);
  }
  
  //xin: 定义总计时事件
  cudaEvent_t start_all, stop_all;
  float time_gpu_all;
  cudaEventCreate(&start_all);
  cudaEventCreate(&stop_all);
  cudaEventRecord(start_all, 0);
  //

  cudaMalloc((void**)&d_m, 
             matrix_dim*matrix_dim*sizeof(float));

  /* beginning of timing point */
  //stopwatch_start(&sw);
  cudaMemcpy(d_m, m, matrix_dim*matrix_dim*sizeof(float), 
	     cudaMemcpyHostToDevice);

  lud_cuda(d_m, matrix_dim);

  cudaMemcpy(m, d_m, matrix_dim*matrix_dim*sizeof(float), 
	     cudaMemcpyDeviceToHost);

  /* end of timing point */
  //stopwatch_stop(&sw);
  //printf("Time consumed(ms): %lf\n", 1000*get_interval_by_sec(&sw));

  cudaFree(d_m);
  
  //xin
  cudaEventRecord(stop_all, 0);
  cudaEventSynchronize(stop_all);
  cudaEventElapsedTime(&time_gpu_all, start_all, stop_all);
  printf("\n=== LUD GPU Total Hardware Time (Figure 7) ===\n");
  printf("Total GPU Time: %f ms\n", time_gpu_all);
  cudaEventDestroy(start_all);
  cudaEventDestroy(stop_all);
  //
  //xin: 内存占用打印 (LUD 是 matrix_dim * matrix_dim 的 float 矩阵)
  // xin: 1. 计算字节数
    unsigned long matrix_size = (unsigned long)matrix_dim * matrix_dim * sizeof(float);
    
    // xin: 2. 换算为 MB (使用 1024.0 * 1024.0 确保浮点精度)
    double matrix_mb = (double)matrix_size / (1024.0 * 1024.0);
    double total_mb = matrix_mb; 
    printf("DEBUG: Application Size = %d x %d nodes\n", matrix_dim, matrix_dim);
    //printf("DEBUG: HtoD Buffer Size = %f MB (%lu bytes)\n", matrix_mb, matrix_size);
    //printf("DEBUG: DtoH Buffer Size = %f MB (%lu bytes)\n", matrix_mb, matrix_size);
    printf("DEBUG: Total Memory Movement = %.2f MB\n", total_mb);
    //

  if (do_verify){
    printf("After LUD\n");
    // print_matrix(m, matrix_dim);
    printf(">>>Verify<<<<\n");
    lud_verify(mm, m, matrix_dim); 
    free(mm);
  }

  free(m);

  return EXIT_SUCCESS;
}				/* ----------  end of function main  ---------- */
