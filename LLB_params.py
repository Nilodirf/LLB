# next step: construct an array of exchange couplings reflecting the sample construction
# also: make a function out of the J_mat definition with input (materials, sample). Generalize it so that a square matrix of sum_i^(n-1) 1  is constructed and must take as many exchange couplings as input
import numpy as np
from scipy import constants as sp
from scipy import interpolate as ip
from matplotlib import pyplot as plt

# In this file, the dynamical parameters needed for the quantum LLB implementation are computed.
# We use mean field theory to get the temperature dependence of these parameters. The scaling is mostly defined by the mean field magnetization at a given temperature.
# As an input for the computations, I assume 1d numpy-arrays of
#   (A1) magnetization_amplitudes m_amp,
#   (A2) angles gamma in the transverse plane m_gamma,
#   (A3) angles phi in the magnetization plane m_phi,
#   (B) the materials (unit cells) with UNIFORM parameters that shall be investigated
#   (C) some structure that defines the configuration of different unit cells
#   (D),(E) electron (phonon)- temperatures Te(p) (at each time step 1 of these 1d arrays is needed)

# (A1,2,3) Let's initialize an array of length N=150 for all the magnetization data:

m_amp=np.ones(150)
m_phi=np.zeros(150)
m_gamma=np.ones(150)*90.

# and define a function to get the magnetization vector:
def get_mag(amp, gamma, phi):
    # This function takes as input parameters the amplitude and angles (A1, A2, A3) and puts out a numpy array of dimension 3xN
    # with 3 magnetization components for N unit cells

    mx=cos(gamma)*cos(phi)
    my=sin(gamma)*cos(phi)
    mz=sin(phi)
    m=np.array([mx, my, mz])
    return(np.array([amp*comp for comp in m]))

# (B) Let's assume a sample consisting of two different materials stacked on top of each other in any arrangement.
# To assign different material parameters for the different layers, I'll define a sample class here:

class material():
    def __init__(self, S, Tc):
        self.S=S                    # effective spin
        self.Tc=Tc                  # Curie temperature

    # With these parameters we can already define the spin-dependent exchange coupling constant for a given material in mean field approximation:
    def exch_const(self):
        J= 3*self.S/(self.S+1)*sp.k*self.Tc
        return(J)


# (C) Let's define a dummy function to construct a sample consisting of different materials of grainsize and their exchange couplings:

def get_sample():
    # This is a dummy function that should definitely be replaced by outputs from your code. It does not take any input parameters as I define everything here.
    # As an output we get
    #   (i) a 1d array of M materials within the sample (materials on the scale of the grainsize of the macrospins)
    #   (ii) a 1d array of the actual sample consisting of stacked layers of the M materials
    #   (iii) an MxM matrix with exchange coupling constants  between different materials

    # Define define three dummy materials with different parameters:
    mat_1 = material(1/2, 650)
    mat_2 = material(7/2, 1200)
    mat_3 = material(1, 300)
    materials=np.array([mat_1, mat_2, mat_3])

    # Define a sample structure where 5 layers of each sam build blocks that are periodically stacked 10 times (5*3*10=150=N):

    building_block=np.concatenate((np.array([mat_1 for _ in range(5)]), np.array([mat_2 for _ in range(5)])))
    sample=np.concatenate([building_block for _ in range(20)])


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
        for j in range(i-1):
            J_mat[i][j]=J_mat[j][i]


# Now the sample is defined on the scale needed for LLB computation. On the basis of this structure we can define for instance the exchange coupling of neighbouring grains:
def exch_coup(sample, mag,):
    # This function takes as input parameters:
    #   (A) the instantaneous magnetization retrieved from get_mag
    # As an output we get an array of exchange coupling for at each site (unit cell) of length len(m_amp)

    J=3*S/(S+1)*sp.k*Tc
    J_on_site=J*m_amp
    J_last_site=

def mean_mag_map(S, Tc):
    # This function takes as input parameters:
    #   (i) the effective spin of the material and
    #   (ii) the Curie temperature of the material
    # As an output we get an interpolation function of the mean field magnetization at any temperature T<=T_c (this can of course be extended to T>T_c with zeros)


def Brillouin(S,Tc, m_amp, Te):
    # This function takes the same input parameters as mean_mag_map. Additionally, we take as an input arrays of
    #   (iii) magnetization length and
    #   (iv) (electron) temperature
    # As an output we get the Brillouin function for the given (iii), (iv) arrays

    # First, define the ratio of magnetic and thermal energies from (iii) and (iv):
    eta=

    c1 = (2 * S + 1) / (2 * S)
    c2 = 1 / (2 * S)
    fb = c1 / np.tanh(c1 * eta) - c2 / np.tanh(c2 * eta)
    return (fb)