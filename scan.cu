


__global__ void Get_Transfer_Flags_Kernel( part_int_t n_total, int side,  Real d_min, Real d_max, Real *pos_d, bool *transfer_flags_d ){
  
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if ( tid >= n_total ) return;
  
  bool transfer = 0;
  
  Real pos = pos_d[tid];
  // if ( tid < 1 ) printf( "%f\n", pos);
  
  if ( side == 0 ){
    if ( pos < d_min ) transfer = 1;
  }
  
  if ( side == 1 ){
    if ( pos >= d_max ) transfer = 1;
  }
  
  // if ( transfer ) printf( "##Thread particles transfer\n");
  
  transfer_flags_d[tid] = transfer;  
}


__global__ void Scan_Kernel( part_int_t n_total, bool *transfer_flags_d, int *prefix_sum_d, int *prefix_sum_block_d ){
  
  __shared__ int data_sh[SCAN_SHARED_SIZE];
  
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


__global__ void Prefix_Sum_Blocks_Kernel( int n_partial, int *prefix_sum_block_d ){
  
  int tid_block, val,  start_index, n_threads;
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



__global__ void Sum_Blocks_Kernel( part_int_t n_total,  int *prefix_sum_d, int *prefix_sum_block_d ){
  
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


__global__ void Get_N_Transfer_Particles_and_Transfer_Last( part_int_t n_total, int *n_transfer_d, bool *transfer_flags_d, int *prefix_sum_d, int *transfer_last_particle_d ){
  n_transfer_d[0] = prefix_sum_d[n_total-1] + (int)transfer_flags_d[n_total-1];
  // if (n_transfer_d[0] != 0 ) printf( "##Thread transfer: %d\n", n_transfer_d[0]); 
  transfer_last_particle_d[0] = (int)transfer_flags_d[n_total-1];
}


__global__ void Get_Transfer_Indices_Kernel( part_int_t n_total, bool *transfer_flags_d, int *prefix_sum_d, int *transfer_indices_d ){
  
  int tid, transfer_index;
  tid =  threadIdx.x + blockIdx.x * blockDim.x;
  if ( tid >= n_total ) return;
  transfer_index = prefix_sum_d[tid];
    
  if ( transfer_flags_d[tid] ) transfer_indices_d[transfer_index] = tid;  
  
}


__global__ void Select_Indices_to_Replace_Tranfered( part_int_t n_total, int n_transfer, int transfer_last, bool *transfer_flags_d, int *prefix_sum_d, int *replace_indices_d ){
  
  int tid, tid_inv;  
  tid = threadIdx.x + blockIdx.x * blockDim.x;
  if ( tid >= n_total ) return;
  tid_inv = n_total - tid - 1;
  
  bool transfer_flag = transfer_flags_d[tid];
  if ( transfer_flag ) return;
  
  int prefix_sum_inv, replace_id;
  
  prefix_sum_inv = n_transfer - prefix_sum_d[tid];
  replace_id = tid_inv - prefix_sum_inv;
  replace_indices_d[replace_id] = tid;
  
}
  
   
   
  