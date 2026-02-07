rom triqs_dft_tools.sumk_dft import *
from triqs.gf import *
from triqs.archive import HDFArchive
from triqs.operators.util import *
from triqs_cthyb import *
import triqs.utility.mpi as mpi

with HDFArchive('CoBr.wf.h5','a') as ar:
    G_tau=ar['dmft_output']['G_tau']
    G_0=ar['dmft_output']['G_0']
    G_iw=ar['dmft_output']['G_iw']
    Sigma_iw=ar['dmft_output']['Sigma_iw']

    # Set the double counting:
    dm =G_iw.density() # compute the density matrix of the impurity problem

    #if mpi.is_master_node(): print "Density matrix DMFT iteration:",iteration_number, "is", dm
    if mpi.is_master_node(): print "occupancy of d1 orbital:",(dm['up_0'][0][0]+dm['down_0'][0][0]).real
    if mpi.is_master_node(): print "occupancy of d2 orbital:",(dm['up_1'][0][0]+dm['down_1'][0][0]).real
    if mpi.is_master_node(): print "occupancy of d3 orbital:",(dm['up_2'][0][0]+dm['down_2'][0][0]).real
    if mpi.is_master_node(): print "occupancy of d4 orbital:",(dm['up_3'][0][0]+dm['down_3'][0][0]).real
    if mpi.is_master_node(): print "occupancy of d5 orbital:",(dm['up_4'][0][0]+dm['down_4'][0][0]).real
