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
block_size = 1024
block1D = ( block_size, 1,  1)



print( "\nCompiling CUDA code" )
cudaCodeFile = open("scan.cu","r")
cudaCodeString = cudaCodeFile.read()
cudaCodeString = cudaCodeString.replace( "TPB_PARTICLES",  str(block_size) )
cudaCodeString = cudaCodeString.replace( "SCAN_SHARED_SIZE",  str(block_size*2) )
cudaCodeString = cudaCodeString.replace( "Real",  "double" )
cudaCodeString = cudaCodeString.replace( "part_int_t",  "int" )
cudaCode = SourceModule(cudaCodeString)
Get_Transfer_Flags_Kernel = cudaCode.get_function('Get_Transfer_Flags_Kernel')
Scan_Kernel = cudaCode.get_function('Scan_Kernel')
Prefix_Sum_Blocks_Kernel = cudaCode.get_function('Prefix_Sum_Blocks_Kernel')
Sum_Blocks_Kernel = cudaCode.get_function('Sum_Blocks_Kernel')
Get_N_Transfer_Particles_and_Transfer_Last_Kernel = cudaCode.get_function('Get_N_Transfer_Particles_and_Transfer_Last')
Get_Transfer_Indices_Kernel = cudaCode.get_function('Get_Transfer_Indices_Kernel')
Select_Indices_to_Replace_Tranfered_Kernel = cudaCode.get_function('Select_Indices_to_Replace_Tranfered')
# Remove_Transfred_Particles_Kernel = cudaCode.get_function('Remove_Transfred_Particles_Kernel')

max_particles = 256**3
n_iterations = 10

for i in range( n_iterations ):

  # n_particles = 255
  n_particles = np.int( np.random.randint(10, max_particles, 1)[0] )
  print( f'\nN particles: {n_particles}' )

  n_transfer_d = gpuarray.to_gpu(np.zeros(1).astype(np.int32))
  transfer_last_d = gpuarray.to_gpu(np.zeros(1).astype(np.int32))
  pos_d = gpuarray.to_gpu( np.zeros(max_particles).astype(np.float64) )
  transfer_flags_d = gpuarray.to_gpu( np.zeros(max_particles).astype(np.bool) )
  grid_size_half_max = ( (max_particles-1)//2   ) // block_size + 1
  prefix_sum_d = gpuarray.to_gpu( np.zeros(max_particles).astype(np.int32) )
  prefix_sum_block_d = gpuarray.to_gpu( np.zeros(grid_size_half_max).astype(np.int32) )
  transfer_indices_d = gpuarray.to_gpu(np.zeros(max_particles).astype(np.int32))
  replace_indices_d = gpuarray.to_gpu(np.zeros(max_particles).astype(np.int32))

  success = True


  grid_size_half = ( (n_particles-1)//2   ) // block_size + 1
  grid_size = ( n_particles - 1 ) // block_size + 1
  grid1D_half = ( grid_size_half, 1, 1 )
  grid1D = ( grid_size, 1, 1 )


  pos_all = np.zeros(max_particles)
  # pos = np.zeros(n_particles)
  # pos[::3] = 2
  pos = np.random.rand( n_particles ) * 1.1
  pos_all[:n_particles] = pos.copy()
  pos_h = pos_all.copy()
  pos_d.set(pos_all)

  d_min = 0
  d_max = 1

  transfer_flags_h = pos >= d_max
  transfer_flags_h_all = np.zeros( max_particles).astype(np.bool)
  transfer_flags_h_all[:n_particles] = transfer_flags_h
  transfer_indices_h = np.where( pos_all >= d_max)[0]
  n_transfer_h = len(transfer_indices_h)
  transfer_last_h = np.int32( transfer_flags_h[-1] )
  sum_cpu = np.zeros(max_particles)
  sum_cpu[1:n_particles] = np.cumsum(transfer_flags_h)[:-1]
  
  # replace_indices_h = [ ]
  # index = 0
  # while len( replace_indices_h ) < n_transfer_h :
  #   id = n_particles - index - 1
  #   if not transfer_flags_h[id]: replace_indices_h.append( id )
  # replace_indices_h = np.array( replace_indices_h, dtype=np.int32 )
  
  replace_indices_h = np.where( transfer_flags_h == False )[0]
  replace_indices_h = replace_indices_h[::-1]
  replace_indices_h = replace_indices_h[:n_transfer_h]
  
  Get_Transfer_Flags_Kernel( np.int32( n_particles), np.int32(1), np.float64(d_min), np.float64(d_max), pos_d, transfer_flags_d, grid=grid1D, block=block1D )
  transfer_flags_0 = transfer_flags_d.get()
  diff_flags = np.abs( transfer_flags_h_all.astype(np.int32) - transfer_flags_0.astype(np.int32) ).sum()
  print( f' Diff flags: {diff_flags}',  )
  if diff_flags != 0: success = False
  
  Scan_Kernel( np.int32(n_particles), transfer_flags_d, prefix_sum_d, prefix_sum_block_d, grid=grid1D_half, block=block1D )
  Prefix_Sum_Blocks_Kernel(np.int32(grid_size_half), prefix_sum_block_d, grid=(1,1,1), block=block1D)
  Sum_Blocks_Kernel( np.int32(n_particles) , prefix_sum_d, prefix_sum_block_d,  grid=grid1D, block=block1D )
  
  sum_gpu = prefix_sum_d.get()
  diff_sum = np.abs( sum_cpu - sum_gpu ).sum()
  print (' Diff sum: ', diff_sum )
  if diff_sum > 1e-10: success = False
  
  Get_N_Transfer_Particles_and_Transfer_Last_Kernel( np.int32(n_particles),  n_transfer_d, transfer_flags_d, prefix_sum_d, transfer_last_d, grid=(1,1,1), block=(1,1,1))
  n_transfer_gpu = n_transfer_d.get()[0]
  transfer_last_gpu = transfer_last_d.get()[0]
  if transfer_last_h != transfer_last_gpu:
    success = False
    print( ' Failed: Transfer last ')
  
  diff_transfer = np.abs( n_transfer_gpu - n_transfer_h )
  print( ' Diff transfer: ', diff_transfer )
  if diff_transfer != 0: success = False
  
  Get_Transfer_Indices_Kernel( np.int32(n_particles) , transfer_flags_d, prefix_sum_d, transfer_indices_d,   grid=grid1D, block=block1D )
  transfer_indices = transfer_indices_d.get()[:n_transfer_gpu]
  
  diff_transf_indx = np.abs( transfer_indices - transfer_indices_h ).sum()
  print( ' Diff transfer indx: ', diff_transf_indx )
  if diff_transf_indx != 0: success = False
  
  
  print( " N transfer: ", n_transfer_gpu )
  
  Select_Indices_to_Replace_Tranfered_Kernel( np.int32(n_particles) ,  n_transfer_gpu, transfer_last_gpu, transfer_flags_d, prefix_sum_d, replace_indices_d,   grid=grid1D, block=block1D )
  replace_indices_gpu = replace_indices_d.get()[:n_transfer_gpu]
  diff_replace = np.abs( replace_indices_gpu - replace_indices_h ).sum()
  print( ' Diff replace indx: ', diff_replace )
  if diff_replace != 0: success = False
  
  if success :  print( 'SUCCES' )
  else:  break


















  # Remove_Transfred_Particles_Kernel( np.int32(n_particles), n_transfer_d, transfer_flags_d, prefix_sum_d, transfer_indices_d, pos_d, grid=(1,1,1), block=block1D)
  # pos_new_d = pos_d.get()[:n_particles-n_transfer]
  # 
  # diff_pos_sum = np.abs( pos_new_d.sum() - pos_new_h.sum())
  # print ' Diff pos sum: ', diff_pos_sum
  # if diff_pos_sum > 1e-9: success = False
  # 
  # 
  # 
