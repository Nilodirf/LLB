# next step: complete get_mean_mag()
# also: make a function out of the J_mat definition with input (materials, sample). Generalize it so that a square matrix of sum_i^(n-1) 1  is constructed and must take as many exchange couplings as input
import numpy as np
from scipy import constants as sp
from scipy import interpolate as ip
from matplotlib import pyplot as plt

# In this file, the dynamical parameters needed for the quantum LLB implementation are computed.
# We use mean field theory to get the temperature dependence of these parameters. The scaling is mostly defined by the mean field magnetization at a given temperature.
# As an input for the computations, I assume 1d numpy-arrays of
#   (1) magnetization_amplitudes m_amp,
#   (2) angles gamma in the transverse plane m_gamma,
#   (3) angles phi in the magnetization plane m_phi,
#   (4) the materials (unit cells) with UNIFORM parameters that shall be investigated
#   (5) some structure that defines the configuration of different grains for magnetization computation
#   (D),(E) electron (phonon)- temperatures Te(p) (at each time step 1 of these 1d arrays is needed)


# (1,2,3,4,5) Let's define a dummy function to construct a sample consisting of different materials (this does not mean atoms but rather a definition for grainsize of at least a unit cell!)
# For this, I construct a class that holds relevant information for the quantum LLB:
class material():
    def __init__(self, name, S, Tc):
        self.name=name                          # name of the material used for the string representation of the class
        self.S=S                                # effective spin
        self.Tc=Tc                              # Curie temperature
        self.J=3*self.S/(self.S+1)*sp.k*self.Tc # mean field exchange coupling constant

    def __str__(self):
        return self.name

    # From only these few parameters we can already abstract the mean field magnetization map, which is the basis of all temperature dependent parameters in the quantum LLB
    def get_mean_mag_map(self):
        # This function computes the mean field mean magnetization map by solving the self-consistent equation m=B(m)
        # As an output we get an interpolation function of the mean field magnetization at any temperature T<=T_c (this can of course be extended to T>T_c with zeros)

        # Start by defining a grid of magnetization amplitude between 0 and 1:
        m_amp_grid = np.arange(0, 1, 1e-5)

        # Then we also need a temperature grid. I'll make it course grained for low temperatures (0.7*Tc) (small slope) and fine grained for large temperatures (large slope):
        temp_grid=np.array(list(np.arange(0, 0.7*self.Tc, 1e-1))+list(np.arange(0.7*self.Tc, self.Tc, 1e-5)))

        # Now call the Brillouin function for the defined grids:
        brillouin=self.get_Brillouin(m_amp_grid, temp_grid)

    def get_Brillouin(self, m_amp, Te):
        # This function takes the same input parameters as mean_mag_map. Additionally, we take as an input arrays of
        #   (iii) magnetization length and
        #   (iv) (electron) temperature
        # As an output we get the Brillouin function as a 2d-array of dimension len(m_amp)xlen(Te) for the given (iii), (iv) arrays

        # First, define the ratio of magnetic and thermal energies from (iii) and (iv):
        eta = self.J*m_amp/sp.k/Te[:np.newaxis]

        c1 = (2 * self.S + 1)/(2 * self.S)
        c2 = 1/(2 * self.S)
        bri_func = c1/np.tanh(c1 * eta)-c2/np.tanh(c2 * eta)
        return bri_func

def get_sample():
    # This is a dummy function that should definitely be replaced by outputs from your code. It does not take any input parameters as I define everything here.
    # As an output we get
    #   (i) a 1d list of M materials within the sample (materials on the scale of the grainsize of the macrospins)
    #   (ii) a 1d numpy array of the actual sample consisting of stacked layers of the M materials
    #   (iii-v) magnetization amplitude and angles

    # Define define three dummy materials with different parameters:
    mat_1 = material('uno', 1/2, 650)
    mat_2 = material('dos', 7/2, 1200)
    mat_3 = material('tres', 1, 300)
    materials=[mat_1, mat_2, mat_3]

    # Define a sample structure where 5 layers of each sam build blocks that are periodically stacked 10 times (5*3*10=150=N):
    building_block=np.concatenate((np.array([mat_1 for _ in range(5)]), np.array([mat_2 for _ in range(5)]), np.array([mat_3 for _ in range(5)])))
    sample=np.concatenate([building_block for _ in range(10)])

    #Define initial magnetization on the whole sample (for simplicity uniform) and fully magnetized along the z-axis
    m_amp = np.ones(150)
    m_phi = np.zeros(150)
    m_gamma = np.zeros(150)
    return materials, sample, m_amp, m_phi, m_gamma


# Now I define a function to get the magnetization vector from amplitude and angles:
def get_mag(amp, gamma, phi):
    # This function takes as input parameters the amplitude and angles (A1, A2, A3) and puts out a numpy array of dimension 3xN
    # with 3 magnetization components for N unit cells

    mx=amp*cos(gamma)*sin(phi)
    my=amp*sin(gamma)*sin(phi)
    mz=amp*cos(phi)
    return(mx, my, mz)

# From the sample the materials within, one can set up an array of exchange-coupling constants between all grains:
def get_exch_coup_sam(materials, sample):
    # This function takes as input parameters:
    #   (i) the 1d-list of magnetic unique materials in the sample (size M)
    #   (ii) the 1d numpy array of the sample, consisting of a material (from class material) at each grain (size N)
    # As an output we get a 2d numpy array of dimension Nx2 for coupling each site with its 2 neighbours in the linear chain of grains. I will define this here:
    ex_coup_arr=np.zeros((len(sample),2))

    # Define a matrix J_mat of dimension len(materials)xlen(materials) with the exchange coupling constants of mat_i and mat_j at J_mat[i][j]=J_mat[j][i]
    J_mat=np.zeros((len(materials), len(materials)))
    # fill the diagonal with the mean field exchange constant of each material:
    for i, mat in enumerate(materials):
        J_mat[i][i]=mat.exch_const()
    # define the off-diagonals, namely some values for exchange coupling constants of different materials:
    J_mat[0][1]=1e-20
    J_mat[1][2]=5e-20
    J_mat[0][2]=1e-19
    # symmetrize the matrix so that also elements [i][j] with i>j can be read out:
    for i in range(1,len(materials)):
        for j in range(i):
            J_mat[i][j]=J_mat[j][i]

    # Now we can assign the coupling of each grain to its nearest neighbours by filling the output array with the respective matrix entry:
    for i, grain in enumerate(sample):
        this_mat_index=materials.index(grain)
        if i>0:
            last_mat_index=materials.index(sample[i-1])
            ex_coup_arr[i][0]=J_mat[this_mat_index][last_mat_index]
        if i<len(sample)-1:
            next_mat_index=materials.index(sample[i+1])
            ex_coup_arr[i][1]=J_mat[this_mat_index][next_mat_index]

    return ex_coup_arr

materials, sample, m_amp, m_phi, m_gamma=get_sample()
exch_coup_const=get_exch_coup_sam(materials, sample)


