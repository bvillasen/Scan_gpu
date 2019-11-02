import sys, time, os
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
#import pycuda.curandom as curandom
from pycuda.elementwise import ElementwiseKernel
from pycuda.reduction import ReductionKernel
import h5py as h5
import matplotlib.pyplot as plt

from cudaTools import setCudaDevice, getFreeMemory, gpuArray3DtocudaArray, kernelMemoryInfo



#Set parameters 

x_min = 0.0
x_max = 50.0
L = x_max - x_min
dx = 5.0


#initialize pyCUDA context
usingAnimation = False
useDevice = 0
cudaDevice = setCudaDevice( devN=useDevice, usingAnimation=usingAnimation)

#set thread grid for CUDA kernels
block_size = 128
block1D = ( block_size, 1,  1)



print "\nCompiling CUDA code"
cudaCodeFile = open("scan.cu","r")
cudaCodeString = cudaCodeFile.read()
cudaCodeString = cudaCodeString.replace( "SHARED_SIZE",  str(block_size*2) )
cudaCodeString = cudaCodeString.replace( "TPB_PARTICLES",  str(block_size) )
cudaCode = SourceModule(cudaCodeString)
Scan_Kernel = cudaCode.get_function('Scan_Kernel')
Sum_Blocks_Kernel = cudaCode.get_function('Sum_Blocks_Kernel')
Get_Transfer_Indexs_Kernel = cudaCode.get_function('Get_Transfer_Indexs_Kernel')
Get_N_Transfer_Particles_Kernel = cudaCode.get_function('Get_N_Transfer_Particles_Kernel')
Remove_Transfred_Particles_Kernel = cudaCode.get_function('Remove_Transfred_Particles_Kernel')
Get_Transfer_Flags_Kernel = cudaCode.get_function('Get_Transfer_Flags_Kernel')
Prefix_Sum_Kernel = cudaCode.get_function('Prefix_Sum_Kernel')


max_particles = 1280
# prefix_sum_block_d = gpuarray.to_gpu( np.zeros(grid_size_half).astype(np.int32) )

# 
for i in range(100):
  
  n_particles = np.random.randint(10, max_particles, 1)[0]
  # n_particles = 10
  # if n_particles%2 == 1 : n_particles += 1
  print '\nN particles: ', n_particles
  
  grid_size_half_max = ( (max_particles-1)/2   ) // block_size + 1
  pos_d = gpuarray.to_gpu( np.zeros(max_particles).astype(np.float64) )
  transfer_flags_d = gpuarray.to_gpu( np.zeros(max_particles).astype(np.bool) )
  prefix_sum_d = gpuarray.to_gpu( np.zeros(max_particles).astype(np.int32) )
  prefix_sum_block_d = gpuarray.to_gpu( np.zeros(grid_size_half_max).astype(np.int32) )
  transfer_indxs_d = gpuarray.to_gpu(np.zeros(max_particles).astype(np.int32))
  n_transfer_d = gpuarray.to_gpu(np.zeros(1).astype(np.int32))


  


  success = True


  grid_size_half = ( (n_particles-1)/2   ) // block_size + 1
  grid_size = ( n_particles - 1 ) // block_size + 1
  grid1D_half = ( grid_size_half, 1, 1 )
  grid1D = ( grid_size, 1, 1 )


  pos_all = np.zeros(max_particles)
  pos = np.zeros(n_particles)
  pos[::3] = 2
  # pos = np.random.rand( n_particles ) + 0.5
  pos_all[:n_particles] = pos.copy()
  pos_h = pos_all.copy()
  
  pos_d.set(pos_all)





  d_min = 0
  d_max = 1

  transfer_flags_h = pos >= d_max
  transfer_indxs_h = np.where( pos_h >= d_max )[0]
  n_transfer_h = transfer_flags_h.sum()
  transfer_flags_h_all = np.zeros( max_particles).astype(np.bool)
  transfer_flags_h_all[:n_particles] = transfer_flags_h
  transfer_indxs_h = np.where( pos_all >= d_max)[0]
  
  n_transfer_h = len(transfer_indxs_h)
  
  pos_new_h = pos_h[ pos_h < d_max]
  # for i in range( )
  
  
  sum_cpu = np.zeros(max_particles)
  sum_cpu[1:n_particles] = np.cumsum(transfer_flags_h)[:-1]
  



  # print np.where(pos < d_max )




  Get_Transfer_Flags_Kernel( np.int32( n_particles), np.int32(1), np.float64(d_min), np.float64(d_max), pos_d, transfer_flags_d, grid=grid1D, block=block1D )
  transfer_flags_0 = transfer_flags_d.get()
  
  diff_flags_0 = np.abs( transfer_flags_h_all.astype(np.int32) - transfer_flags_0.astype(np.int32) ).sum()
  print ' Diff flags: ', diff_flags_0
  if diff_flags_0 != 0: success = False
  
  # 
  Scan_Kernel( np.int32(n_particles), transfer_flags_d, prefix_sum_d, prefix_sum_block_d, grid=grid1D_half, block=block1D )
  # prefix_sum_block = prefix_sum_block_d.get()
  # prefix_sum_block_sum = np.zeros(grid_size_half_max).astype(np.int32)
  # prefix_sum_block_sum[1:grid_size_half] = np.cumsum(prefix_sum_block[:grid_size_half])[:-1]
  # prefix_sum_block_d.set(prefix_sum_block_sum )
  Prefix_Sum_Kernel(np.int32(grid_size_half), prefix_sum_block_d, grid=(1,1,1), block=block1D)
  
  Sum_Blocks_Kernel( np.int32(n_particles) , prefix_sum_d, prefix_sum_block_d,  grid=grid1D, block=block1D )
  sum_gpu = prefix_sum_d.get()
  
  diff_sum = np.abs( sum_cpu - sum_gpu ).sum()
  print ' Diff sum: ', diff_sum
  if diff_sum > 1e-10: success = False
  

  transfer_flags_1 = transfer_flags_d.get()
  diff_flags_1 = np.abs( transfer_flags_h_all.astype(np.int32) - transfer_flags_1.astype(np.int32) ).sum()
  print ' Diff flags: ', diff_flags_1
  if diff_flags_1 != 0: success = False
  
  
  Get_N_Transfer_Particles_Kernel( np.int32(n_particles),  n_transfer_d, transfer_flags_d, prefix_sum_d, grid=(1,1,1), block=(1,1,1))
  n_transfer = n_transfer_d.get()[0]
  
  diff_transfer = np.abs( n_transfer - n_transfer_h )
  print ' Diff transfer: ', diff_transfer
  if diff_transfer != 0: success = False
  
  
  Get_Transfer_Indexs_Kernel( np.int32(n_particles) , transfer_flags_d, prefix_sum_d, transfer_indxs_d,   grid=grid1D, block=block1D )
  transfer_indxs = transfer_indxs_d.get()[:n_transfer]
  
  diff_transf_indx = np.abs( transfer_indxs - transfer_indxs_h ).sum()
  print ' Diff transfer indx: ', diff_transf_indx
  if diff_transf_indx != 0: success = False
  
  print " N transfer: ", n_transfer
  Remove_Transfred_Particles_Kernel( np.int32(n_particles), n_transfer_d, transfer_flags_d, prefix_sum_d, transfer_indxs_d, pos_d, grid=(1,1,1), block=block1D)
  pos_new_d = pos_d.get()[:n_particles-n_transfer]
  
  diff_pos_sum = np.abs( pos_new_d.sum() - pos_new_h.sum())
  print ' Diff pos sum: ', diff_pos_sum
  if diff_pos_sum > 1e-9: success = False



  if ( success ):
    print 'SUCCES'
  else:
    
    break
  