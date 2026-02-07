from triqs_dft_tools.sumk_dft import *
from triqs.gf import *
from h5 import HDFArchive
from triqs.operators.util import *
from triqs.operators.util.U_matrix import *
from triqs.operators.util.hamiltonians import *
from triqs_cthyb import *
#from triqs.plot.mpl_interface import *
import sys, triqs.version as triqs_version
import triqs.utility.mpi as mpi
import numpy as np

dft_filename = 'LiFeAs'          # filename
U = 6                        # interaction parameters
J = 0.5 
beta = 40                       # inverse temperature
loops = 25                     # number of DMFT loops
mix = 0.5                       # mixing factor of Sigma after solution of the AIM
dc_type = 1                     # DC type: 0 FLL, 1 Held, 2 AMF
use_blocks = False               # use bloc structure from DFT input
prec_mu = 0.001                # precision of chemical potential
d=0

SK = SumkDFT(hdf_file=dft_filename+'.h5',use_dft_blocks=False, beta=beta )

Sigma = SK.block_structure.create_gf(beta=beta)

SK.put_Sigma([Sigma])

G = SK.extract_G_loc(transform_to_solver_blocks=False)

SK.analyse_block_structure_from_gf(G, threshold=0.01)





'''
SK = SumkDFT(hdf_file=dft_filename+'.h5',use_dft_blocks=False, beta=beta)

Sigma = SK.block_structure.create_gf(beta=beta)
SK.put_Sigma([Sigma])
G = SK.extract_G_loc(transform_to_solver_blocks=False)
SK.analyse_block_structure(threshold=0.01)
'''

if mpi.is_master_node():
    SK.calculate_diagonalization_matrix(prop_to_be_diagonal='eal', calc_in_solver_blocks=True)

SK.block_structure=mpi.bcast(SK.block_structure)
SK.block_structure.approximate_as_diagonal()


'''
SK.calculate_diagonalization_matrix(prop_to_be_diagonal='eal', calc_in_solver_blocks=True)
SK.block_structure.approximate_as_diagonal()
'''

for i_sh in range(len(SK.deg_shells)):
    num_block_deg_orbs = len(SK.deg_shells[i_sh])
    mpi.report('found {0:d} blocks of degenerate orbitals in shell {1:d}'.format(num_block_deg_orbs, i_sh))
    for iblock in range(num_block_deg_orbs):
        mpi.report('block {0:d} consists of orbitals:'.format(iblock))
        for keys in list(SK.deg_shells[i_sh][iblock].keys()):
            mpi.report('  '+keys)

#Sigma = SK.block_structure.create_gf(beta=beta)
#SK.put_Sigma([Sigma])
#G = SK.extract_G_loc()
#SK.analyse_block_structure_from_gf(G, threshold=0.01)


#if mpi.is_master_node():
#    SK.calculate_diagonalization_matrix(prop_to_be_diagonal='eal', calc_in_solver_blocks=True)

#SK.block_structure=mpi.bcast(SK.block_structure)
#SK.block_structure.approximate_as_diagonal()

p = {}
# solver
p["random_seed"] = 123 * mpi.rank + 567
p["length_cycle"] = 200
p["n_warmup_cycles"] = 75000
p["n_cycles"] = 750000
# tail fit
#p["perform_tail_fit"] = False
p["perform_tail_fit"] = True
p["fit_max_moment"] = 4
p["fit_min_n"] = 40
p["fit_max_n"] = 80
p['imag_threshold']= 1e-9
'''
gm = {}
gm['flip_spins'] = {("up_0",0) : ("down_0",0), ("down_0",0) : ("up_0",0), ("up_1",0) : ("down_1",0), ("down_1",0) : ("up_1",0), ("up_2",0) : ("down_2",0), ("down_2",0) : ("up_2",0), ("up_3",0) : ("down_3",0), ("down_3",0) : ("up_3",0), ("up_4",0) : ("down_4",0), ("down_4",0) : ("up_4",0)}
p['move_global'] = gm
p['move_global_prob'] = 0.5

p["move_global"] = {
    "SpinFlip" : {
        ("up_0"  ,0) : ("down_0",0),
        ("up_1"  ,0) : ("down_1",0),
        ("up_2"  ,0) : ("down_2",0),
        ("down_0",0) : ("up_0"  ,0),
        ("down_1",0) : ("up_1"  ,0),
        ("down_2",0) : ("up_2"  ,0)
    }
}
'''
n_orb = SK.corr_shells[0]['dim']
l = SK.corr_shells[0]['l']
spin_names = ["up","down"]
orb_names = [i for i in range(n_orb)]
# Use GF structure determined by DFT blocks:
gf_struct = [(block, indices) for block, indices in SK.gf_struct_solver[0].items()]

#gf_struct = SK.block_structure.gf_struct_solver_list[0]

print("\n")
mpi.report('Sumk to Solver: %s'%SK.sumk_to_solver)
print("\n")
mpi.report('GF struct sumk: %s'%SK.gf_struct_sumk)
print("\n")
mpi.report('GF struct solver: %s'%SK.gf_struct_solver)

# Construct U matrix for density-density calculations:
Umat, Upmat = U_matrix_kanamori(n_orb=n_orb, U_int=U, J_hund=J)


#h_int = h_int_density(spin_names, orb_names, map_operator_structure=SK.sumk_to_solver[0], U=Umat, Uprime=Upmat)
h_int=h_int_kanamori(spin_names, orb_names, U=Umat, Uprime=Upmat, J_hund=J, off_diag=None, map_operator_structure=SK.sumk_to_solver[0])
#h_int  = diagonal_part(SK.block_structure.convert_operator(h_sumk))
S = Solver(beta=beta, gf_struct=gf_struct)

previous_runs = 0
previous_present = False
'''
if mpi.is_master_node():
	with HDFArchive(dft_filename+'.h5','a') as ar:
     		ar.create_group('dmft_output')

if mpi.is_master_node():
    f = HDFArchive(dft_filename+'.h5','a')
    if 'dmft_output' in f:
        ar = f['dmft_output']
        if 'iterations' in ar:
            previous_present = True
            previous_runs = ar['iterations']
    else:
        f.create_group('dmft_output')
    del f
'''    

for iteration_number in range(1,loops+1):
    if mpi.is_master_node(): print ("Iteration = ", iteration_number)

    #SK.symm_deg_gf(S.Sigma_iw)                        # symmetrizing Sigma
    SK.set_Sigma([ S.Sigma_iw ])                            # put Sigma into the SumK class
    chemical_potential = SK.calc_mu( precision = prec_mu )  # find the chemical potential for given density
    S.G_iw << SK.extract_G_loc()[0]                         # calc the local Green function
    if mpi.is_master_node(): mpi.report("Total charge of Gloc : %.6f"%S.G_iw.total_density())

    # Init the DC term and the real part of Sigma, if no previous runs found:
    if (iteration_number==1 and previous_present==False):
        dm = S.G_iw.density()
        SK.calc_dc(dm, U_interact = U, J_hund = J, orb = 0, use_dc_formula = dc_type)
       # S.Sigma_iw << SK.dc_imp[0]['up'][0,0]
        
        S.Sigma_iw['up_0'][0,0] << SK.dc_imp[0]['up'][0,0] + d
        S.Sigma_iw['down_0'][0,0] << SK.dc_imp[0]['down'][0,0] - d
        S.Sigma_iw['up_1'][0,0] << SK.dc_imp[0]['up'][1,1] + d
        S.Sigma_iw['down_1'][0,0] << SK.dc_imp[0]['down'][1,1] - d
        S.Sigma_iw['up_2'][0,0] << SK.dc_imp[0]['up'][2,2] + d
        S.Sigma_iw['up_3'][0,0] << SK.dc_imp[0]['up'][3,3] + d
        S.Sigma_iw['up_4'][0,0] << SK.dc_imp[0]['up'][4,4] + d
        S.Sigma_iw['down_2'][0,0] << SK.dc_imp[0]['down'][2,2] - d
        S.Sigma_iw['down_3'][0,0] << SK.dc_imp[0]['down'][3,3] - d
        S.Sigma_iw['down_4'][0,0] << SK.dc_imp[0]['down'][4,4] - d
        
    # Calculate new G0_iw to input into the solver:
    S.G0_iw << S.Sigma_iw + inverse(S.G_iw)
    S.G0_iw << inverse(S.G0_iw)

    # Solve the impurity problem:
    S.solve(h_int=h_int, **p)
    
    S.Sigma_iw['up_0'] << (S.Sigma_iw['up_0'] + S.Sigma_iw['down_0'])/2
    S.Sigma_iw['down_0'] << S.Sigma_iw['up_0']   
    
    S.Sigma_iw['up_1'] << (S.Sigma_iw['up_1'] + S.Sigma_iw['down_1'])/2
    S.Sigma_iw['down_1'] << S.Sigma_iw['up_1']

    S.Sigma_iw['up_2'] << (S.Sigma_iw['up_2'] + S.Sigma_iw['down_2'])/2
    S.Sigma_iw['down_2'] << S.Sigma_iw['up_2']

    S.Sigma_iw['up_3'] << (S.Sigma_iw['up_3'] + S.Sigma_iw['down_3'])/2
    S.Sigma_iw['down_3'] << S.Sigma_iw['up_3']

    S.Sigma_iw['up_4'] << (S.Sigma_iw['up_4'] + S.Sigma_iw['down_4'])/2
    S.Sigma_iw['down_4'] << S.Sigma_iw['up_4']


    # Solved. Now do post-solution stuff:
    if mpi.is_master_node: mpi.report("Total charge of impurity problem : %.6f"%S.G_iw.total_density())

    # Now mix Sigma and G with factor mix, if wanted:
    if (iteration_number>1 or previous_present):
        if mpi.is_master_node():
            with HDFArchive(dft_filename+'.h5','r') as ar:
                mpi.report("Mixing Sigma and G with factor %s"%mix)
                S.Sigma_iw << mix * S.Sigma_iw + (1.0-mix) * ar['dmft_output']['Sigma_iw']
                S.G_iw << mix * S.G_iw + (1.0-mix) * ar['dmft_output']['G_iw']
        S.G_iw << mpi.bcast(S.G_iw)
        S.Sigma_iw << mpi.bcast(S.Sigma_iw)

    # Write the final Sigma and G to the hdf5 archive:
    if mpi.is_master_node():
        with HDFArchive(dft_filename+'.h5','a') as ar:
              ar['dmft_output']['iterations'] = iteration_number
              ar['dmft_output']['G_0'] = S.G0_iw
              ar['dmft_output']['G_tau'] = S.G_tau
              ar['dmft_output']['G_iw'] = S.G_iw
              ar['dmft_output']['Sigma_iw'] = S.Sigma_iw

    # Set the new double counting:
    dm = S.G_iw.density() # compute the density matrix of the impurity problem
    occ_up=dm['up_0'][0][0]+dm['up_1'][0][0]+dm['up_2'][0][0]+dm['up_3'][0][0]+dm['up_4'][0][0]
    occ_dn=dm['down_0'][0][0]+dm['down_1'][0][0]+dm['down_2'][0][0]+dm['down_3'][0][0]+dm['down_4'][0][0]
    wannier_moment=occ_up-occ_dn
    mpi.report("Wannier Moment : %.6f"%wannier_moment.real)

    SK.calc_dc(dm, U_interact = U, J_hund = J, orb = 0, use_dc_formula = dc_type)

    # Save stuff into the user_data group of hdf5 archive in case of rerun:
    SK.save(['chemical_potential','dc_imp','dc_energ'])
