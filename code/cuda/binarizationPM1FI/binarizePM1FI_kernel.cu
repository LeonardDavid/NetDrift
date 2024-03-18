#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdio.h>
#include <cstdint>

#include <curand.h>
#include <curand_kernel.h>
#include <chrono>

#define DEBUG_1D 0
#define DEBUG_THREAD_INFO_FLOAT32 0
#define DEBUG_THREAD_INFO_INT32 0
#define DEBUG_BITS 0
#define DEBUG_SEEDS 0

// bit stuff https://stackoverflow.com/questions/111928/is-there-a-printf-converter-to-print-in-binary-format

// for rng stuff: http://ianfinlayson.net/class/cpsc425/notes/cuda-random
template <typename scalar_t>
__global__ void binarizePM1FI_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> input,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> output,
    float f01,
    float f10,
    unsigned long long seed0,
    int block_size,
    double* d_index_offset_flat,
    int row_size
  ) {

  // handle access indices
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int d = blockIdx.y * blockDim.y + threadIdx.y;
  const int e = blockIdx.z * blockDim.z + threadIdx.z;

  if ((c < input.size(0)) && (d < input.size(1)) && (e < input.size(2)))
  {
    // curandState_t state0;
    // curand_init(seed0, 0, 0, &state0);
    // curand_init(clock() + c, 0, 0, &state0);

    float i_used = input[c][d][e];

    // if(c==0 && d==0 && e==0){
    //   printf("%f ", d_index_offset_flat[c]);
    // }

    // printf("%d\n", block_size);
    size_t index = c*row_size + blockIdx.y;
    // int bound = d - blockIdx.y*block_size - d_index_offset_flat[index];
    if(0 <= d - d_index_offset_flat[index] && d_index_offset_flat[index] < row_size*block_size){
    // if(0 <= bound && bound < block_size-1){
      // in legal bounds, read with offset (can be 0 -> read actual value)
      i_used = input[c][d-d_index_offset_flat[index]][e];
    }
    else{
      // out of bounds (at edge of row), read random value
      // if(c%2==0)
      //   i_used = 1;
      // else
      //   i_used = -1;
      i_used = c%2 == 0 ? 1 : -1;
    }

    output[c][d][e] = i_used;
  }
}

torch::Tensor binarizePM1FI_cuda(
  torch::Tensor input,
  float f01,
  float f10,
  std::vector<std::vector<double>> index_offset,
  float block_size
) {
  // The number of thread blocks in a grid is usually dictated by the size of the data being processed, which typically exceeds the number of processors in the system.
  // dim3 threadsPerBlock(8,8,8)
  // <<<number of blocks per grid, number of threads ber block>>>
  // grid is created with enough blocks to have one thread per matrix element

  // https://devtalk.nvidia.com/default/topic/1028226/how-many-concurrent-threads-are-running-on-my-geforce-gtx-1080-ti-/

  
  torch::Tensor output = input;
  // print(input);
  // std::cout<<"before:"<<std::endl;
  // print(input[0]);

  // std::cout<<"cuda_index_offset:"<<std::endl;
  // for(int i = 0; i<index_offset.size(); i++){
  //   for(int j = 0; j<index_offset[i].size();j++){
  //     std::cout<<index_offset[i][j]<<" ";
  //   }
  //   std::cout<<std::endl;
  // }
  
  // useful values
  int row_size = index_offset[0].size();
  int total_size = index_offset.size()*index_offset[0].size();
  // std::cout<<index_offset.size()<<"x"<<index_offset[0].size()<<"="<<total_size<<std::endl;

  // flatten index_offset matrix for gpu
  double* index_offset_flat = new double[total_size];
  int index = 0;
  for(int i = 0; i<index_offset.size(); i++){
    for(int j = 0; j<index_offset[i].size();j++){
      index_offset_flat[index] = index_offset[i][j];
      index++;
    }
  }

  // std::cout<<"flat_index_offset:"<<std::endl;
  // for(int i = 0; i<total_size; i++){
  //   std::cout<<index_offset_flat[i]<<" ";
  // }
  // std::cout<<std::endl;

  int64_t shape_len = input.dim();
  std::vector<int64_t> shape_original;
  for (int i = 0; i < shape_len; i++)
  {
    shape_original.push_back(input.size(i));
  }

  if (shape_len == 1)
  {
    input = input.reshape({input.size(0),1,1});
    output = output.reshape({output.size(0),1,1});
  }
  if (shape_len == 2)
  {
    input = input.reshape({input.size(0),input.size(1),1});
    output = output.reshape({output.size(0),output.size(1),1});
  }
  if (shape_len > 3)
  {
    input = input.reshape({input.size(0),input.size(1),-1});
    output = output.reshape({output.size(0),output.size(1),-1});
  }

  const int input_size_x = input.size(0);
  const int input_size_y = input.size(1);
  const int input_size_z = input.size(2);

  int threads_x = 1; // per block, 8
  // TODO maybe actually should be block_size??
  int threads_y = 64; // per block, 8
  int threads_z = 1; // per block, 8

  #if DEBUG_1D
    threads_x = 1;
    threads_y = 1;
    threads_z = 1;
  #endif

  const dim3 threads(threads_x,threads_y, threads_z);
  const dim3 blocks((input_size_x + threads_x - 1) / threads_x,
                    (input_size_y + threads_y - 1) / threads_y,
                    (input_size_z + threads_z - 1) / threads_z);

  // create a seed from the current time in nanoseconds
  auto now = std::chrono::system_clock::now();
  auto now_ms = std::chrono::time_point_cast<std::chrono::nanoseconds>(now);
  auto value = now_ms.time_since_epoch();
  unsigned long long seed0 = value.count();

  //manually copy index_offset from cpu to gpu using cudaMalloc and cudaMemcpy
  double* d_index_offset_flat;
  cudaMalloc((void **) &d_index_offset_flat, total_size*sizeof(double));
  cudaMemcpy(d_index_offset_flat, index_offset_flat, (total_size*sizeof(double)), cudaMemcpyHostToDevice);
  
  AT_DISPATCH_ALL_TYPES(input.type(), "binarizePM1FI_cuda", ([&] {
    binarizePM1FI_kernel<scalar_t><<<blocks, threads>>>(
        input.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        output.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        f01,
        f10,
        seed0,
        int(block_size),
        d_index_offset_flat,
        row_size
    );
  }));

  input = output.reshape(shape_original);
  // std::cout<<"after:"<<std::endl;
  // print(input[0]);

  cudaFree(d_index_offset_flat);
  delete index_offset_flat;

  return input;
}
