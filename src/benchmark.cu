#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "cuda_helper.hpp"

using namespace std;

template <typename T>
bool EqualAarrays(const vector<T> &a1, const vector<T> &a2)
{
  size_t n = a1.size();
  if (n != a2.size())
  {
    return false;
  }
  for (size_t i = 0; i < n; ++i)
  {
    if (a1[i] != a2[i])
    {
      cout << "invalid i is " << i << " a1[i]" << a1[i] << " a2[i]" << a2[i] << endl;
      return false;
    }
  }
  return true;
}

template <>
bool EqualAarrays(const vector<half> &a1, const vector<half> &a2)
{
  size_t n = a1.size();
  if (n != a2.size())
  {
    return false;
  }
  for (size_t i = 0; i < n; ++i)
  {
    if ((float)a1[i] != (float)a2[i])
    {
      cout << "invalid i is " << i << " a1[i]" << (float)a1[i] << " a2[i]" << (float)a2[i] << endl;
      return false;
    }
  }
  return true;
}

template <typename T>
__global__ void CopyScalarKernel(T *d_in, T *d_out, const size_t n)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < n; i += stride)
  {
    d_out[i] = d_in[i];
  }
}

template <typename T>
void CopyScalar(T *d_in, T *d_out, size_t n)
{
  int max_blocks = 4096;
  int threads = 1024;
  int blocks = min((int)(n + threads - 1) / threads, max_blocks);
  CopyScalarKernel<T><<<blocks, threads>>>(d_in, d_out, n);
}

template <typename T>
__global__ void CopyVector1Kernel(T *d_in, T *d_out, const size_t n)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const float ratio = ((float)sizeof(int)) / sizeof(T);
  const int m = n / ratio;
  int i;
  for (i = idx; i < m; i += stride)
  {
    reinterpret_cast<int *>(d_out)[i] = reinterpret_cast<int *>(d_in)[i];
  }
}

template <typename T>
void CopyVector1(T *d_in, T *d_out, size_t n)
{
  int threads = 1024;
  int max_blocks = 4096;
  const float ratio = ((float)sizeof(int)) / sizeof(T);
  int blocks = min((int)(n / ratio + threads - 1) / threads, max_blocks);
  CopyVector1Kernel<T><<<blocks, threads>>>(d_in, d_out, n);
}

template <typename T>
__global__ void CopyVector2Kernel(T *d_in, T *d_out, const size_t n)
{
  const float ratio = ((float)sizeof(int2)) / sizeof(T);
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const int m = n / ratio;
  for (int i = idx; i < m; i += stride)
  {
    reinterpret_cast<int2 *>(d_out)[i] = reinterpret_cast<int2 *>(d_in)[i];
  }
}

template <typename T>
void CopyVector2(T *d_in, T *d_out, size_t n)
{
  int threads = 1024;
  int max_blocks = 4096;
  const float ratio = ((float)sizeof(int2)) / sizeof(T);
  int blocks = min((int)(n / ratio + threads - 1) / threads, max_blocks);
  CopyVector2Kernel<T><<<blocks, threads>>>(d_in, d_out, n);
}

template <typename T>
__global__ void CopyVector4Kernel(T *d_in, T *d_out, const size_t n)
{
  const float ratio = ((float)sizeof(int4)) / sizeof(T);
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const int m = n / ratio;
  for (int i = idx; i < m; i += stride)
  {
    reinterpret_cast<int4 *>(d_out)[i] = reinterpret_cast<int4 *>(d_in)[i];
  }
}

template <typename T>
void CopyVector4(T *d_in, T *d_out, size_t n)
{
  int threads = 1024;
  int max_blocks = 4096;
  const float ratio = ((float)sizeof(int4)) / sizeof(T);
  int blocks = min((int)(n / ratio + threads - 1) / threads, max_blocks);
  CopyVector4Kernel<T><<<blocks, threads>>>(d_in, d_out, n);
}

template <typename T>
void RunFunc(int func_id, T *d_in, T *d_out, size_t array_size)
{
  switch (func_id)
  {
  case 0:
    CopyScalar<T>(d_in, d_out, array_size);
    break;
  case 1:
    CopyVector1<T>(d_in, d_out, array_size);
    break;
  case 2:
    CopyVector2<T>(d_in, d_out, array_size);
    break;
  case 3:
    CopyVector4<T>(d_in, d_out, array_size);
    break;
  default:
    assert(false);
  }
}

template <typename T>
void VectrizedMemoryAccessBenchmark(int func_id)
{
  T *d_in = nullptr;
  T *d_out = nullptr;
  int n_trials = 10;
  int log_min_array_size = 10;
  int log_max_array_size = 31;
  for (int log_array_size = log_min_array_size; log_array_size < log_max_array_size; ++log_array_size)
  {
    size_t array_size = 1 << log_array_size;
    size_t array_byte_size = sizeof(T) * array_size;
    vector<T> h_in(array_size);
    vector<T> h_out(array_size);
    for (int i = 0; i < array_size; ++i)
    {
      h_in[i] = i;
      h_out[i] = 0;
    }
    checkCudaErrors(cudaMalloc(&d_in, array_byte_size));
    checkCudaErrors(cudaMalloc(&d_out, array_byte_size));
    checkCudaErrors(cudaMemcpy(d_in, h_in.data(), array_byte_size, cudaMemcpyDefault));

    // dummy run;
    RunFunc(func_id, d_in, d_out, array_size);
    cudaDeviceSynchronize();
    chrono::system_clock::time_point start = chrono::system_clock::now();
    for (int trial_i = 0; trial_i < n_trials; ++trial_i)
    {
      RunFunc(func_id, d_in, d_out, array_size);
    }
    cudaDeviceSynchronize();
    chrono::system_clock::time_point end = chrono::system_clock::now();
    double elapsed_time = static_cast<double>(chrono::duration_cast<chrono::microseconds>(end - start).count()) / n_trials * 1e-6;
    double throughput = array_byte_size / elapsed_time;
    cout << "func_id\t" << func_id << "\tarray_byte_size\t" << array_byte_size << "\ttime:\t" << elapsed_time << "\tsec.\tthroughput:\t" << throughput * 1e-9 << "\tGB/s" << endl;
    checkCudaErrors(cudaMemcpy(h_out.data(), d_out, array_byte_size, cudaMemcpyDefault));
    assert(EqualAarrays(h_in, h_out));

    // free device memory
    checkCudaErrors(cudaFree(d_in));
    d_in = nullptr;
    checkCudaErrors(cudaFree(d_out));
    d_out = nullptr;
  }
}

int main(int argc, char *argv[])
{
  cout << "half" << endl;
  for (int func_id = 1; func_id < 4; ++func_id)
  {
    VectrizedMemoryAccessBenchmark<half>(func_id);
  }
  cout << "float" << endl;
  for (int func_id = 0; func_id < 4; ++func_id)
  {
    VectrizedMemoryAccessBenchmark<float>(func_id);
  }
  cout << "double" << endl;
  for (int func_id = 0; func_id < 4; ++func_id)
  {
    if (func_id == 1)
    {
      continue;
    }
    VectrizedMemoryAccessBenchmark<double>(func_id);
  }
}