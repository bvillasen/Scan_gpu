


__global__ void Get_Transfer_Flags_Kernel( int n_total, int side,  double d_min, double d_max, double *pos_d, bool *transfer_flags_d ){
  
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if ( tid >= n_total ) return;
  
  bool transfer = 0;
  
  double pos = pos_d[tid];
  
  if ( side == 0 ){
    if ( pos < d_min ) transfer = 1;
  }
  
  if ( side == 1 ){
    if ( pos >= d_max ) transfer = 1;
  }
  
  transfer_flags_d[tid] = transfer;  
}

__global__ void Scan_Kernel( int n_total, bool *transfer_flags_d, int *prefix_sum_d, int *prefix_sum_block_d ){
  
  __shared__ int data_sh[SHARED_SIZE];
  
  int tid_block, block_start;
  // tid = threadIdx.x + blockIdx.x * blockDim.x;
  tid_block = threadIdx.x;
  
  block_start = 2*blockIdx.x*blockDim.x; 
  
  data_sh[2*tid_block] = block_start + 2*tid_block < n_total ? (int) transfer_flags_d[block_start + 2*tid_block]  :  0;
  data_sh[2*tid_block+1] = block_start + 2*tid_block+1 < n_total ?  (int) transfer_flags_d[block_start + 2*tid_block+1]  :  0;
  __syncthreads();
  
  int offset = 1;
  int n = blockDim.x*2;
  
  int ai, bi;
  int t;
  
  for (int d = n/2; d>0; d/=2){
  
    __syncthreads();
    if ( tid_block < d ){
      ai = offset*(2*tid_block+1)-1;
      bi = offset*(2*tid_block+2)-1;
      data_sh[bi] += data_sh[ai];
    }
    offset *= 2;
  }
  
  // Clear the last element
  if (tid_block == 0) data_sh[n - 1] = 0;  
  
  // Traverse down tree & build scan
  for (int d = 1; d < n; d *= 2){
  
    __syncthreads();
    offset /=2;
    if (tid_block < d){
  
      ai = offset*(2*tid_block+1)-1;
      bi = offset*(2*tid_block+2)-1;
  
      t = data_sh[ai];
      data_sh[ai] = data_sh[bi];
      data_sh[bi] += t; 
    }
  }
  __syncthreads();
  
  // Write results to device memory
  if ( block_start + 2*tid_block < n_total )  prefix_sum_d[block_start + 2*tid_block] = data_sh[2*tid_block]; 
  if ( block_start + 2*tid_block+1 < n_total) prefix_sum_d[block_start + 2*tid_block+1] = data_sh[2*tid_block+1];
  
  // Write the block sum 
  int last_flag_block = (int) transfer_flags_d[block_start + 2*(blockDim.x-1)+1];
  if (tid_block == 0) prefix_sum_block_d[blockIdx.x] = data_sh[2*(blockDim.x-1)+1] + last_flag_block;
}


__global__ void Prefix_Sum_Kernel( int n_partial, int *prefix_sum_block_d ){
  
  int tid_block, val, counter, start_index, n_threads;
  tid_block = threadIdx.x;
  n_threads = blockDim.x;
  
  __shared__ int data_sh[TPB_PARTICLES];
  
  
  int sum = 0;
  int n = 0;
  start_index = n * n_threads;
  while( start_index < n_partial ){
    data_sh[tid_block] = start_index+tid_block < n_partial  ?  prefix_sum_block_d[start_index+tid_block] :  0;
    __syncthreads();
    
    
    if (tid_block == 0){
      for ( int i=0; i<n_threads; i++ ){
        val = data_sh[i];
        data_sh[i] = sum;
        sum += val;
      }
    }
    __syncthreads();
    
    if (start_index + tid_block < n_partial) prefix_sum_block_d[start_index+tid_block] = data_sh[tid_block];
    n += 1;
    start_index = n * n_threads;
    
  }
}

__global__ void Sum_Blocks_Kernel( int n_total,  int *prefix_sum_d, int *prefix_sum_block_d){
  
  int tid, tid_block, block_id, data_id;
  tid = threadIdx.x + blockIdx.x * blockDim.x;
  tid_block = threadIdx.x;
  block_id = blockIdx.x;
  data_id = block_id/2;
  
  __shared__ int block_sum_sh[1];
  
  if ( tid_block == 0 ){
    block_sum_sh[0] = prefix_sum_block_d[data_id];
    // printf( "%d   %d\n",  block_id/2, prefix_sum_block[data_id] );
  }
  __syncthreads();
   
  
  if (tid < n_total) prefix_sum_d[tid] += block_sum_sh[0];
  
}

__global__ void Get_Transfer_Indexs_Kernel( int n_total, bool *transfer_flags_d, int *prefix_sum_d, int *transfer_indxs_d){
  
  int tid, transfer_index;
  tid =  threadIdx.x + blockIdx.x * blockDim.x;
  if ( tid >= n_total ) return;
  transfer_index = prefix_sum_d[tid];
    
  if ( transfer_flags_d[tid] ) transfer_indxs_d[transfer_index] = tid;  
  
}

__global__ void Get_N_Transfer_Particles_Kernel( int n_total, int *n_transfer_d, bool *transfer_flags_d, int *prefix_sum_d ){
  n_transfer_d[0] = prefix_sum_d[n_total-1] + (int)transfer_flags_d[n_total-1];
}

// 
// __global__ void Remove_Transfred_Particles_Kernel( int n_total, int *n_transfered_d, bool *transfer_flags_d, int *prefix_sum_d, int *transfer_indxs_d, double *data_d ){
// 
//   int tid_block, dst_indx ;
//   tid_block =  threadIdx.x;
//   // n_threads = blockDim.x;
// 
//   int n_replace, n_transfered, n_replaced;
//   n_transfered = n_transfered_d[0];
//   n_replace = n_transfered;
// 
// 
//   n_replaced = 0;
// 
//   if (tid_block == 0 ){
//     for ( int i=n_total-1; i>=0; i--){
//       if ( !transfer_flags_d[i] ){
//         if ( n_replaced == n_replace ) break;
//         dst_indx = transfer_indxs_d[n_replaced];
//         data_d[dst_indx] = data_d[i];
//         n_replaced += 1;
//       }
//     }
//   }
// }
  
__global__ void Remove_Transfred_Particles_Kernel( int n_total, int *n_transfered_d, bool *transfer_flags_d, int *prefix_sum_d, int *transfer_indxs_d, double *data_d ){

  int tid_block, start_index, n, n_threads ;
  tid_block =  threadIdx.x;
  n_threads = blockDim.x;
  int dst_indx;

  int n_replace, n_transfered;
  n_transfered = n_transfered_d[0];
  // n_replace = n_transfered > ( n_total - n_transfered)  ?  n_total - n_transfered  : n_transfered;
  n_replace = n_transfered;

  __shared__ int N_replaced_sh[1];
  __shared__ bool transfer_flags_sh[TPB_PARTICLES];

  if ( tid_block == 0 ) N_replaced_sh[0] = 0;
  __syncthreads();
// 
  if (tid_block == 0 ) printf( " N replace: %d \n", n_replace );
// 
  n = 1;
  while( N_replaced_sh[0] < n_replace ){
    // if (n==100) break;
    // if (tid_block == 0 ) printf(" Iteration: %d\n", n );
    start_index =  n_total - n*n_threads;
    if ( start_index + n_threads < 0 ) break;
    if ( start_index + tid_block >= 0 && start_index + tid_block < n_total) transfer_flags_sh[tid_block] = transfer_flags_d[start_index + tid_block];
    // else  transfer_flags_sh[tid_block] = 1; 
    __syncthreads();
    // printf( "%d \n", n*n_threads );
  // 
    // if (tid_block == 0 ){
    //   for ( int i=n_threads-1; i>=0; i--){
    //     if ( start_index + i >= n_total || start_index + i < 0 ) continue;
    //     printf("%d    %d     %d \n", start_index + i, (int)transfer_flags_d[start_index + i], (int)transfer_flags_sh[i]  );
    //     // if (transfer_flags_d[start_index + i] != transfer_flags_sh[i] && start_index + i < n_total) printf("ERROR\n");
    //   }
    // }
    // 
    if ( tid_block == 0 ){
      for ( int i=n_threads-1; i>=0; i--){
        if ( start_index + i >= n_total || start_index + i < 0 ){
          // printf("Error in deleting tranfered particle data \n" );
          continue;
        }
        // printf( "%d  %d\n", start_index + i, (int)transfer_flags_sh[i]  );
        // if ( !transfer_flags_sh[i] ){
        if ( !transfer_flags_d[start_index+i] ){
          if ( N_replaced_sh[0] == n_replace ) continue;
          dst_indx = transfer_indxs_d[N_replaced_sh[0]];
          printf("moving  %d   to   %d   %f  ->  %f  %d   %d  n_replaced: %d\n", start_index + i, dst_indx, data_d[start_index + i], data_d[dst_indx], (int) transfer_flags_d[start_index + i], (int) transfer_flags_sh[i], N_replaced_sh[0] +1 );
          data_d[dst_indx] = data_d[start_index + i];
          N_replaced_sh[0] += 1;
          __syncthreads();
        }
      }
      // printf("%d\n",n );
    }
    n += 1;
    __syncthreads();
  }


  if ( tid_block == 0 ) printf(" N iterations: %d\n", n );


}

  
  
  
  // 
  
  
  
  


// __global__ void Sum_Blocks_Kernel( int n_total, int n_per_block, int n_blocks, int *output, int *output_block){
// 
//   int tid, tid_block;
//   tid = threadIdx.x + blockIdx.x * blockDim.x;
//   tid_block = threadIdx.x;
// 
//   extern __shared__ float data_sh[];
//   while( tid_block < n_blocks ){
//     data_sh[tid_block] = output_block[tid_block];
//     tid_block += blockDim.x;
//   }
// 
//   tid_block = threadIdx.x;
//   if ( tid_block == 0 ){
//     for (int i=1; i<blockIdx.x/2; i++){
//       data_sh[0] += data_sh[i];
//     }
//   }
//   __syncthreads();
// 
// 
//   if ( blockIdx.x/2 > 0 ) output[tid] += data_sh[0];
// 
// 
// 
// }

















// 
// __global__ void prescan(float *g_odata, float *g_idata, int n)
// {
//   extern __shared__ float temp[];  // allocated on invocation
//   int thid = threadIdx.x;
//   int offset = 1;
// 
// 
//   temp[2*thid] = g_idata[2*thid]; // load input into shared memory
//   temp[2*thid+1] = g_idata[2*thid+1];
// 
//   for (int d = n>>1; d > 0; d >>= 1)                    // build sum in place up the tree
//   { 
//     __syncthreads();
//     if (thid < d)
//     {
// 
//       int ai = offset*(2*thid+1)-1;
//       int bi = offset*(2*thid+2)-1;
// 
// 
//         temp[bi] += temp[ai];
//     }
//     offset *= 2;
//   }
// 
// 
// if (thid == 0) { temp[n - 1] = 0; } // clear the last element
// 
// 
// for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
// {
//      offset >>= 1;
//      __syncthreads();
//      if (thid < d)                     
//      {
// 
// 
//     int ai = offset*(2*thid+1)-1;
//     int bi = offset*(2*thid+2)-1;
// 
// 
// float t = temp[ai];
// temp[ai] = temp[bi];
// temp[bi] += t; 
//       }
// }
//  __syncthreads();
// 
//      g_odata[2*thid] = temp[2*thid]; // write results to device memory
//      g_odata[2*thid+1] = temp[2*thid+1];
// 
// }
// 








// __global__ void scan(float *g_odata, float *g_idata, int n)
// {
  //   extern __shared__ float temp[]; // allocated on invocation
  //   int thid = threadIdx.x;
  //   int pout = 0, pin = 1;
  //   // Load input into shared memory.
  //   // This is exclusive scan, so shift right by one
  //   // and set first element to 0
  //   temp[pout*n + thid] = (thid > 0) ? g_idata[thid-1] : 0;
  //   __syncthreads();
  //   for (int offset = 1; offset < n; offset *= 2)
  //   {
    //      pout = 1 - pout; // swap double buffer indices
    //      pin = 1 - pout;
    //      if (thid >= offset)
    //        temp[pout*n+thid] += temp[pin*n+thid - offset];
    //      else
    //        temp[pout*n+thid] = temp[pin*n+thid];
    //      __syncthreads();
    //     }
    //     g_odata[thid] = temp[pout*n+thid]; // write output
    //     }
















