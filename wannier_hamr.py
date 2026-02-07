import numpy as np
  
data=np.loadtxt(open('E_mu_nu.wf_hr.dat','r'))
wannierargs=np.where((data[:,0]==0) & (data[:,1]==0)&(data[:,2]==0))
data_reduced= data[wannierargs]
matrix=np.array(data_reduced[:,5])
matrix= matrix.reshape(22,22)
eigen_vals=np.linalg.eig(matrix)
print("The Wannier Hamiltonian is:")
print(matrix)
print("The eigen energies are:")
print(np.sort(eigen_vals[0]))
        
