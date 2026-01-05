long long get_time() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (tv.tv_sec * 1000000) + tv.tv_usec;
}

__global__ void hotspotOpt1(float *p, float* tIn, float *tOut, float sdc,
        int nx, int ny, int nz,
        float ce, float cw, 
        float cn, float cs,
        float ct, float cb, 
        float cc) 
{
    float amb_temp = 80.0;

    int i = blockDim.x * blockIdx.x + threadIdx.x;  
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    int c = i + j * nx;
    int xy = nx * ny;

    int W = (i == 0)        ? c : c - 1;
    int E = (i == nx-1)     ? c : c + 1;
    int N = (j == 0)        ? c : c - nx;
    int S = (j == ny-1)     ? c : c + nx;

    float temp1, temp2, temp3;
    temp1 = temp2 = tIn[c];
    temp3 = tIn[c+xy];
    tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
        + cn * tIn[N] + cb * temp1 + ct * temp3 + sdc * p[c] + ct * amb_temp;
    c += xy;
    W += xy;
    E += xy;
    N += xy;
    S += xy;

    for (int k = 1; k < nz-1; ++k) {
        temp1 = temp2;
        temp2 = temp3;
        temp3 = tIn[c+xy];
        tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
            + cn * tIn[N] + cb * temp1 + ct * temp3 + sdc * p[c] + ct * amb_temp;
        c += xy;
        W += xy;
        E += xy;
        N += xy;
        S += xy;
    }
    temp1 = temp2;
    temp2 = temp3;
    tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
        + cn * tIn[N] + cb * temp1 + ct * temp3 + sdc * p[c] + ct * amb_temp;
    return;
}

void hotspot_opt1(float *p, float *tIn, float *tOut,
        int nx, int ny, int nz,
        float Cap, 
        float Rx, float Ry, float Rz, 
        float dt, int numiter) 
{
    float ce, cw, cn, cs, ct, cb, cc;
    float stepDivCap = dt / Cap;
    ce = cw =stepDivCap/ Rx;
    cn = cs =stepDivCap/ Ry;
    ct = cb =stepDivCap/ Rz;

    cc = 1.0 - (2.0*ce + 2.0*cn + 3.0*ct);

    size_t s = sizeof(float) * nx * ny * nz;  
    float  *tIn_d, *tOut_d, *p_d;

    // xin: 定义 GPU 计时事件
    cudaEvent_t start_total, stop_total;
    float time_gpu_total;
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    cudaEventRecord(start_total, 0);
    //
    cudaMalloc((void**)&p_d,s);
    cudaMalloc((void**)&tIn_d,s);
    cudaMalloc((void**)&tOut_d,s);
    cudaMemcpy(tIn_d, tIn, s, cudaMemcpyHostToDevice);
    cudaMemcpy(p_d, p, s, cudaMemcpyHostToDevice);

    cudaFuncSetCacheConfig(hotspotOpt1, cudaFuncCachePreferL1);

    dim3 block_dim(64, 4, 1);
    dim3 grid_dim(nx / 64, ny / 4, 1);

    //long long start = get_time();
    for (int i = 0; i < numiter; ++i) {
        hotspotOpt1<<<grid_dim, block_dim>>>
            (p_d, tIn_d, tOut_d, stepDivCap, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc);
        float *t = tIn_d;
        tIn_d = tOut_d;
        tOut_d = t;
    }
    cudaDeviceSynchronize();
    //long long stop = get_time();
    //float time = (float)((stop - start)/(1000.0 * 1000.0));
    //printf("Time: %.3f (s)\n",time);
    cudaMemcpy(tOut, tOut_d, s, cudaMemcpyDeviceToHost);
    cudaFree(p_d);
    cudaFree(tIn_d);
    cudaFree(tOut_d);

    // xin: --- 2. GPU 计时终点 ---
    cudaEventRecord(stop_total, 0);
    cudaEventSynchronize(stop_total);
    cudaEventElapsedTime(&time_gpu_total, start_total, stop_total);
    printf("\n=== Hotspot3D GPU Performance Metrics ===\n");
    printf("Total Hardware Time (Init+HtoD+Exec+DtoH+Close): %.3f ms\n", time_gpu_total);
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    //
    //xin 
    // HtoD 阶段需要传输 Power(p) 和 TempIn(tIn) 两个数组
    // DtoH 阶段需要传回 TempOut(tOut) 一个数组
    unsigned long h2d_bytes = (unsigned long)s * 2;
    unsigned long d2h_bytes = (unsigned long)s;
    printf("\n=== Memory Usage Analysis ===\n");
    printf("Grid Size: %d x %d x %d\n", nx, ny, nz);
    //printf("HtoD Transfer Size: %f MB\n", h2d_bytes / (1024.0 * 1024.0));
    //printf("DtoH Transfer Size: %f MB\n", d2h_bytes / (1024.0 * 1024.0));
    printf("Total GPU Memory Allocated: %.2f MB\n", (s * 3) / (1024.0 * 1024.0));
    //

    return;
}

