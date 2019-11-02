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

# prescan_Kernel = cudaCode.get_function('prescan')

# #Initialize Data
# pos = np.random.rand( n_particles ) * (L + 2*dx) - (x_min + dx)
# pos[0] = -1
# transf_flags = pos<x_min
# trans_scan = transf_flags.cumsum()
# 
for i in range(100):
  n_particles = np.random.randint(1290, 13000, 1)[0]
  if n_particles%2 == 1 : n_particles += 1
  # n_particles = 256
  print '\nN particles: ', n_particles


  success = True


  grid_size_half = ( (n_particles-1)/2   ) // block_size + 1
  grid_size = ( n_particles + block_size - 1 ) // block_size
  grid1D_half = ( grid_size_half, 1, 1 )
  grid1D = ( grid_size, 1, 1 )


  # pos = np.linspace(1, n_particles, n_particles)
  pos = np.zeros(n_particles)
  # pos[::2] = 1 
  # pos[::3] = 1 
  pos[::5] = 2 
  pos[-1] = 2

  pos = np.random.rand( n_particles ) + 0.5
  # pos[-1] = 0.5


  pos_h = pos.copy()
  pos_d = gpuarray.to_gpu( pos.astype(np.float64) )


  d_min = 0
  d_max = 1

  transfer_flags_h = pos >= d_max
  transfer_indxs_h = np.where( pos >= d_max )[0]
  n_transfer_h = transfer_flags_h.sum()
  pos_new_h = pos_h[ pos < d_max]
  # for i in range( )

  sum_cpu = np.zeros(n_particles)
  sum_cpu[1:] = np.cumsum(transfer_flags_h)[:-1]


  # print np.where(pos < d_max )




  transfer_flags_d = gpuarray.to_gpu( pos.astype(np.bool) )
  Get_Transfer_Flags_Kernel( np.int32( n_particles), np.int32(1), np.float64(d_min), np.float64(d_max), pos_d, transfer_flags_d, grid=grid1D, block=block1D )
  transfer_flags = transfer_flags_d.get()
  # print pos_d
  # print transfer_flags

  sum_cpu = np.zeros(n_particles).astype(np.int32)
  sum_cpu[1:] = np.cumsum(transfer_flags)[:-1] 

  prefix_sum_d = gpuarray.to_gpu( np.zeros(n_particles).astype(np.int32) )
  prefix_sum_block_d = gpuarray.to_gpu( np.zeros(grid_size_half).astype(np.int32) )
  # 
  Scan_Kernel( np.int32(n_particles), transfer_flags_d, prefix_sum_d, prefix_sum_block_d, grid=grid1D_half, block=block1D )
  prefix_sum_block = prefix_sum_block_d.get()
  prefix_sum_block_sum = np.zeros(grid_size_half).astype(np.int32)
  prefix_sum_block_sum[1:] = np.cumsum(prefix_sum_block)[:-1]
  prefix_sum_block_d = gpuarray.to_gpu(prefix_sum_block_sum.astype(np.int32))
  # 
  Sum_Blocks_Kernel( np.int32(n_particles) , prefix_sum_d, prefix_sum_block_d,  grid=grid1D, block=block1D )
  # prefix_sum_d.set( sum_cpu )
  sum_gpu = prefix_sum_d.get()
  print (pos_h - pos_d.get() ).sum()

  print ( sum_gpu - sum_cpu ).sum()
  # # print sum_gpu 
  # # print sum_cpu
  print (transfer_flags_h.astype(np.int) - transfer_flags_d.get().astype(np.int)).sum()
  print (transfer_flags_h.astype(np.int) - transfer_flags.astype(np.int)).sum()

  transfer_indxs = np.zeros(n_particles).astype(np.int32)
  transfer_indxs_d = gpuarray.to_gpu(transfer_indxs)
  # 
  Get_Transfer_Indexs_Kernel( np.int32(n_particles) , transfer_flags_d, prefix_sum_d, transfer_indxs_d,   grid=grid1D, block=block1D )
  transfer_indxs = transfer_indxs_d.get()
  # 
  n_transfer = np.zeros(1).astype(np.int32)
  n_transfer_d = gpuarray.to_gpu(n_transfer)
  Get_N_Transfer_Particles_Kernel( np.int32(n_particles),  n_transfer_d, transfer_flags_d, prefix_sum_d, grid=(1,1,1), block=(1,1,1))
  n_transfer = n_transfer_d.get()[0]
  # print n_transfer

  # pos_new = pos_d.get()
  # n_total = n_particles
  # # n_replace = n_total - n_transfer if  n_transfer > ( n_total - n_transfer)  else n_transfer
  # n_replace = n_transfer
  # n_replaced = 0
  # for i in range(n_replace):
  #   indx_src = n_total - i - 1
  #   if ( not transfer_flags[indx_src] ):
  #     indx_dst = transfer_indxs[n_replaced]
  #     # print pos_new[indx_src] pos_new[indx_dst], 
  #     pos_new[indx_dst] = pos_new[indx_src]
  #     n_replaced += 1
  # pos_new_d = pos_new[:n_particles-n_transfer]

  # print pos_d
  # print transfer_flags_d
  
  Remove_Transfred_Particles_Kernel( np.int32(n_particles), n_transfer_d, transfer_flags_d, prefix_sum_d, transfer_indxs_d, pos_d, grid=(1,1,1), block=block1D)
  pos_new_d = pos_d.get()[:n_particles-n_transfer]



  diff_pos = np.abs(pos_new_h.sum() - pos_new_d.sum())
  print diff_pos
  if diff_pos > 1e-6: success = False



  if ( success ):
    print 'SUCCES'
  else:

    print "Host:"
    print ' N transfer: ', n_transfer_h
    print " Pos sum: ", pos_new_h.sum()
    

    print '\nDevice:'
    print ' N transfer: ', n_transfer, transfer_flags[-1].astype(np.int32) + prefix_sum_d[-1]
    print ' Pos sum: ', pos_new_d.sum()

    print '\nDifference:'
    print ' N transfer: ', n_transfer_h - n_transfer
    print ' Transfer flags: ', (transfer_flags_h.astype(np.int) - transfer_flags.astype(np.int)).sum()
    print ' Pos sum : ', diff_pos
    print ' Flags Sum: ', (sum_cpu - sum_gpu).sum()
    # print ' Transfer Indx: ', transfer_indxs_h - transfer_indxs[:n_transfer]
    
    break
    
    # pos_new = pos_new[:n_total-n_transfer]
    # pos_new_h.sort()
    # pos_new.sort()
  # print pos_new_h - pos_new
  # break

  # print pos_d
  # print n_transfer
  # print transfer_flags_d, transfer_indxs
  # # 
  # diff = sum_gpu - sum_cpu
  # print (sum_gpu - sum_cpu).min(), (sum_gpu - sum_cpu).max() 



