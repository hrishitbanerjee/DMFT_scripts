from triqs_dft_tools.sumk_dft import *
from triqs.gf import *
from h5 import HDFArchive
from triqs.operators.util import *
from triqs_cthyb import *
import triqs.utility.mpi as mpi

dft_filename = 'LaFeAsOF'          # filename.h5
U = [4, 4 ]                         # interaction parameters
J = [0.3, 0.3]
beta = 20                       # inverse temperature
loops = 25                     # number of DMFT loops
mix = 0.5                       # mixing factor of Sigma after solution of the AIM
dc_type = 1                     # DC type: 0 FLL, 1 Held, 2 AMF
use_blocks = False               # use bloc structure from DFT input
prec_mu = 0.001                # precision of chemical potential
d=0
SK = SumkDFT(hdf_file=dft_filename+'.h5',use_dft_blocks=use_blocks, beta=beta)
if mpi.is_master_node():
    SK.calculate_diagonalization_matrix(prop_to_be_diagonal='eal', calc_in_solver_blocks=True)

SK.block_structure=mpi.bcast(SK.block_structure)
SK.block_structure.approximate_as_diagonal()



p = [{}, {}]
# solver Ir
p[0]["random_seed"] = 143 * mpi.rank + 579
p[0]["length_cycle"] = 200
p[0]["n_warmup_cycles"] = 150000
p[0]["n_cycles"] = 1500000
# tail fit
#p[0]["perform_tail_fit"] = False
p[0]["perform_tail_fit"] = True
p[0]["fit_max_moment"] = 4
p[0]["fit_min_n"] = 30
p[0]["fit_max_n"] = 90
#p[0]["imag_threshold"]=0.001
p[1]=p[0]

n_orb = SK.corr_shells[0]['dim']
l = SK.corr_shells[0]['l']
spin_names = ["up","down"]
orb_names = [i for i in range(n_orb)]
# Use GF structure determined by DFT blocks:
gf_struct = [(block, indices) for block, indices in SK.gf_struct_solver[0].items()]
# Construct U matrix for density-density calculations:

S=[]
h_int=[]
for ii in range(2):
    Umat, Upmat = U_matrix_kanamori(n_orb=n_orb, U_int=U[ii], J_hund=J[ii])
    h_int.append(h_int_density(spin_names, orb_names, map_operator_structure=SK.sumk_to_solver[ii], U=Umat, Uprime=Upmat))
    S.append(Solver(beta=beta, gf_struct=gf_struct))
#Solvers ready!!!!


previous_runs = 0
previous_present = False

if mpi.is_master_node():
    with HDFArchive(dft_filename+'.h5','a') as ar:
        ar.create_group('dmft_output')
'''
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
#assert 0

for iteration_number in range(1,loops+1):
    if mpi.is_master_node(): print ("Iteration = ", iteration_number)

    #if (iteration_number==1 and previous_present==False):
    #    for ii in xrange(2): SK.symm_deg_gf(S[ii].Sigma_iw)        # symmetrizing Sigma for iteration=1
    SK.set_Sigma([ S[0].Sigma_iw, S[1].Sigma_iw ])                            # put Sigma into the SumK class
    chemical_potential = SK.calc_mu( precision = prec_mu )  # find the chemical potential for given density
    Gloc = SK.extract_G_loc()                         # calc the local Green function
    #SK.analyse_block_structure_from_gf(Gloc, threshold=0.1)
    for ii in range(2): S[ii].G_iw << Gloc[ii]
    
    for ii in range(2): mpi.report("Total charge of Gloc : %.6f"%S[ii].G_iw.total_density())

    # Init the DC term and the real part of Sigma, if no previous runs found:
    if (iteration_number==1 and previous_present==False):
        for ii in range(2):
            dm = S[ii].G_iw.density()
            if mpi.is_master_node(): print ("Atom ",ii,": ",dm)
            SK.calc_dc(dm, U_interact = U[ii], J_hund = J[ii], orb = ii, use_dc_formula = dc_type)
            S[ii].Sigma_iw['up_0'][0,0] << SK.dc_imp[SK.inequiv_to_corr[ii]]['up'][0,0] + d
            S[ii].Sigma_iw['down_0'][0,0] << SK.dc_imp[SK.inequiv_to_corr[ii]]['down'][0,0] - d
            S[ii].Sigma_iw['up_1'][0,0] << SK.dc_imp[SK.inequiv_to_corr[ii]]['up'][1,1] + d
            S[ii].Sigma_iw['down_1'][0,0] << SK.dc_imp[SK.inequiv_to_corr[ii]]['down'][1,1] - d
            S[ii].Sigma_iw['up_2'][0,0] << SK.dc_imp[SK.inequiv_to_corr[ii]]['up'][2,2] + d
            S[ii].Sigma_iw['down_2'][0,0] << SK.dc_imp[SK.inequiv_to_corr[ii]]['down'][2,2] - d
            S[ii].Sigma_iw['up_3'][0,0] << SK.dc_imp[SK.inequiv_to_corr[ii]]['up'][3,3] + d
            S[ii].Sigma_iw['down_3'][0,0] << SK.dc_imp[SK.inequiv_to_corr[ii]]['down'][3,3] - d
            S[ii].Sigma_iw['up_4'][0,0] << SK.dc_imp[SK.inequiv_to_corr[ii]]['up'][4,4] + d
            S[ii].Sigma_iw['down_4'][0,0] << SK.dc_imp[SK.inequiv_to_corr[ii]]['down'][4,4] - d

    for ii in range(2):
            # Calculate new G0_iw to input into the solver:
        S[ii].G0_iw << S[ii].Sigma_iw + inverse(S[ii].G_iw)
        S[ii].G0_iw << inverse(S[ii].G0_iw)

            #write intial green's functions
    if mpi.is_master_node():
        with HDFArchive(dft_filename+'.h5','a') as ar:
            ar['dmft_output']['G_0_Ni1']      = S[0].G0_iw
            ar['dmft_output']['G_0_Ni2']      = S[1].G0_iw
        
            # Solve the impurity problem:
    for ii in range(2):    
        S[ii].solve(h_int=h_int[ii], **p[ii])
            # Solved. Now do post-solution stuff:
        if mpi.is_master_node():mpi.report("Total charge of impurity problem %s: %.6f"%(ii,S[ii].G_iw.total_density()))
    '''
    if (iteration_number>1 or previous_present):
        if mpi.is_master_node():
            with HDFArchive(dft_filename+'.h5','r') as ar:
                mpi.report("Mixing Sigma and G with factor %s"%mix)
            S[0].Sigma_iw << mix * S[0].Sigma_iw + (1.0-mix) * ar['dmft_output']['Sigma_iw_Ir1']
            S[1].Sigma_iw << mix * S[1].Sigma_iw + (1.0-mix) * ar['dmft_output']['Sigma_iw_Ir2']
            S[0].G_iw << mix * S[0].G_iw + (1.0-mix) * ar['dmft_output']['G_iw_Ir1']
            S[1].G_iw << mix * S[1].G_iw + (1.0-mix) * ar['dmft_output']['G_iw_Ir2']
            for ii in range(2):
                S[ii].G_iw << mpi.bcast(S[ii].G_iw)
                S[ii].Sigma_iw << mpi.bcast(S[ii].Sigma_iw)
    '''
    # Write the final Sigma and G to the hdf5 archive:
    if mpi.is_master_node():
        with HDFArchive(dft_filename+'.h5','a') as ar:
            ar['dmft_output']['iterations'] = iteration_number
            ar['dmft_output']['G_0_Ni1']      = S[0].G0_iw
            ar['dmft_output']['G_tau_Ni1']    = S[0].G_tau
            ar['dmft_output']['G_iw_Ni1']     = S[0].G_iw
            ar['dmft_output']['Sigma_iw_Ni1'] = S[0].Sigma_iw
            ar['dmft_output']['G_0_Ni2']      = S[1].G0_iw
            ar['dmft_output']['G_tau_Ni2']    = S[1].G_tau
            ar['dmft_output']['G_iw_Ni2']     = S[1].G_iw
            ar['dmft_output']['Sigma_iw_Ni2'] = S[1].Sigma_iw

    for ii in range(2):
            # Set the new double counting:
        dm = S[ii].G_iw.density() # compute the density matrix of the impurity problem
        SK.calc_dc(dm, U_interact = U[ii], J_hund = J[ii], orb = ii, use_dc_formula = dc_type)

    if mpi.is_master_node():
        with HDFArchive(dft_filename+'.h5','a') as ar:
            G_iw_Ni1=ar['dmft_output']['G_iw_Ni1']
            dm_Ni1=G_iw_Ni1.density()
            occ_up_Ni1=dm_Ni1['up_0'][0][0]+dm_Ni1['up_1'][0][0]+dm_Ni1['up_2'][0][0]+dm_Ni1['up_3'][0][0]+dm_Ni1['up_4'][0][0]
            occ_dn_Ni1=dm_Ni1['down_0'][0][0]+dm_Ni1['down_1'][0][0]+dm_Ni1['down_2'][0][0]+dm_Ni1['down_3'][0][0]+dm_Ni1['down_4'][0][0]
            
            G_iw_Ni2=ar['dmft_output']['G_iw_Ni2']
            dm_Ni2=G_iw_Ni2.density()
            occ_up_Ni2=dm_Ni2['up_0'][0][0]+dm_Ni2['up_1'][0][0]+dm_Ni2['up_2'][0][0]+dm_Ni2['up_3'][0][0]+dm_Ni2['up_4'][0][0]
            occ_dn_Ni2=dm_Ni2['down_0'][0][0]+dm_Ni2['down_1'][0][0]+dm_Ni2['down_2'][0][0]+dm_Ni2['down_3'][0][0]+dm_Ni2['down_4'][0][0]

        wannier_moment_Ni1=occ_up_Ni1-occ_dn_Ni1
        wannier_moment_Ni2=occ_up_Ni2-occ_dn_Ni2
        mpi.report("Wannier Moment Ni1: %.6f"%wannier_moment_Ni1.real)
        mpi.report("Wannier Moment Ni2: %.6f"%wannier_moment_Ni2.real)
        # Save stuff into the user_data group of hdf5 archive in case of rerun:
    SK.save(['chemical_potential','dc_imp','dc_energ'])
