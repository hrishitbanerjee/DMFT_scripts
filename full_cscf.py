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
from triqs_dft_tools.converters.wien2k import *

dft_filename = os.getcwd().rpartition('/')[2]
Converter = Wien2kConverter(filename = dft_filename)
Converter.convert_dft_input()

#dft_filename = os.getcwd().rpartition('/')[2]

#dft_filename = 'svo'          # filename
U = 4.0                         # interaction parameters
J = 0.65
beta = 40                       # inverse temperature
loops = 1                      # number of DMFT loops
mix = 0.8                       # mixing factor of Sigma after solution of the AIM
dc_type = 1                     # DC type: 0 FLL, 1 Held, 2 AMF
use_blocks = True               # use bloc structure from DFT input
prec_mu = 0.0001                # precision of chemical potential

SK = SumkDFT(hdf_file=dft_filename+'.h5',use_dft_blocks=use_blocks, beta=beta)

p = {}
# solver
p["random_seed"] = 123 * mpi.rank + 567
p["length_cycle"] = 200
p["n_warmup_cycles"] = 50000
p["n_cycles"] = 500000
# tail fit
p["perform_tail_fit"] = True
p["fit_max_moment"] = 4
p["fit_min_n"] = 30
p["fit_max_n"] = 60

n_orb = SK.corr_shells[0]['dim']
l = SK.corr_shells[0]['l']
spin_names = ["up","down"]
orb_names = [i for i in range(n_orb)]
# Use GF structure determined by DFT blocks:
gf_struct = [(block, indices) for block, indices in SK.gf_struct_solver[0].items()]
# Construct U matrix for density-density calculations:
Umat, Upmat = U_matrix_kanamori(n_orb=n_orb, U_int=U, J_hund=J)


h_int = h_int_density(spin_names, orb_names, map_operator_structure=SK.sumk_to_solver[0], U=Umat, Uprime=Upmat)
S = Solver(beta=beta, gf_struct=gf_struct)

#================================================
# Restarting, important for full CSC:
previous_runs = 0
previous_present = False
if mpi.is_master_node():
    with HDFArchive(dft_filename+'.h5','a') as f:
        if 'dmft_output' in f:
            ar = f['dmft_output']
            if 'iterations' in ar:
                previous_present = True
                previous_runs = ar['iterations']
                print("Start from previous run!")
        else:
            f.create_group('dmft_output')

previous_runs = mpi.bcast(previous_runs)
previous_present = mpi.bcast(previous_present)
     
if previous_present:
    if mpi.is_master_node():
        with HDFArchive(dft_filename+'.h5','r') as ar:
            S.Sigma_iw << ar['dmft_output']['Sigma_iw']

    S.Sigma_iw << mpi.bcast(S.Sigma_iw)
    chemical_potential,dc_imp,dc_energ = SK.load(['chemical_potential','dc_imp','dc_energ'])
    SK.set_mu(chemical_potential)
    SK.set_dc(dc_imp,dc_energ)
#================================================
    

for iteration_number in range(1,loops+1):
    if mpi.is_master_node(): print("Iteration = ", iteration_number)

    SK.symm_deg_gf(S.Sigma_iw)                        # symmetrizing Sigma
    SK.set_Sigma([ S.Sigma_iw ])                            # put Sigma into the SumK class
    chemical_potential = SK.calc_mu( precision = prec_mu )  # find the chemical potential for given density
    S.G_iw << SK.extract_G_loc()[0]                         # calc the local Green function
    mpi.report("Total charge of Gloc : %.6f"%S.G_iw.total_density())

    # Init the DC term and the real part of Sigma, if no previous runs found:
    if (iteration_number==1 and previous_present==False):
        dm = S.G_iw.density()
        SK.calc_dc(dm, U_interact = U, J_hund = J, orb = 0, use_dc_formula = dc_type)
        S.Sigma_iw << SK.dc_imp[0]['up'][0,0]

    # Calculate new G0_iw to input into the solver:
    S.G0_iw << S.Sigma_iw + inverse(S.G_iw)
    S.G0_iw << inverse(S.G0_iw)

    # Solve the impurity problem:
    S.solve(h_int=h_int, **p)

    # Solved. Now do post-solution stuff:
    mpi.report("Total charge of impurity problem : %.6f"%S.G_iw.total_density())

    # Now mix Sigma and G with factor mix, if wanted:
    #if (iteration_number>1 or previous_present):
    #    if mpi.is_master_node():
    #        with HDFArchive(dft_filename+'.h5','r') as ar:
    #            mpi.report("Mixing Sigma and G with factor %s"%mix)
    #            S.Sigma_iw << mix * S.Sigma_iw + (1.0-mix) * ar['dmft_output']['Sigma_iw']
    #            S.G_iw << mix * S.G_iw + (1.0-mix) * ar['dmft_output']['G_iw']
    #    S.G_iw << mpi.bcast(S.G_iw)
    #    S.Sigma_iw << mpi.bcast(S.Sigma_iw)

    # Write the final Sigma and G to the hdf5 archive:
    if mpi.is_master_node():
        with HDFArchive(dft_filename+'.h5','a') as ar:
              ar['dmft_output']['iterations'] = iteration_number + previous_runs
              ar['dmft_output']['G_0'] = S.G0_iw
              ar['dmft_output']['G_tau'] = S.G_tau
              ar['dmft_output']['G_iw'] = S.G_iw
              ar['dmft_output']['Sigma_iw'] = S.Sigma_iw


    # Set the new double counting:
    dm = S.G_iw.density() # compute the density matrix of the impurity problem
    SK.calc_dc(dm, U_interact = U, J_hund = J, orb = 0, use_dc_formula = dc_type)

    # Save stuff into the user_data group of hdf5 archive in case of rerun:
    SK.save(['chemical_potential','dc_imp','dc_energ'])

SK.symm_deg_gf(S.Sigma_iw)
S.G_iw << inverse(S.G0_iw) - S.Sigma_iw
S.G_iw.invert()

# find exact chemical potential
SK.set_Sigma([ S.Sigma_iw ])
chemical_potential = SK.calc_mu( precision = 0.000001 )
dN, d = SK.calc_density_correction(filename = dft_filename+'.qdmft')
SK.save(['chemical_potential','dc_imp','dc_energ'])
correnerg = 0.5 * (S.G_iw * S.Sigma_iw).total_density().real
correnerg -= SK.dc_energ[0]

if mpi.is_master_node():
  f=open(dft_filename+'.qdmft','a')
  f.write("%.16f\n"%correnerg)
  f.close()
