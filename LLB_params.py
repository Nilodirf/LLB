# next step:    (i) find and implement the temperature dependence of interaction terms
#               (ii) check simulation of damping (/rates) with parameters from Unais thesis
# also: make a function out of the J_mat definition with input (materials, sample). Generalize it so that a square matrix of sum_i^(n-1) 1  is constructed and must take as many exchange couplings as input
import numpy as np
from scipy import constants as sp
from scipy import optimize as op
from scipy import interpolate as ip
from matplotlib import pyplot as plt
import itertools
import time

# In this file, the dynamical parameters needed for the quantum LLB implementation are computed.
# We use mean field theory to get the temperature dependence of these parameters. The scaling is mostly defined by the mean field magnetization at a given temperature.
# As an input for the computations, I assume 1d numpy-arrays of
#   (1) magnetization amplitudes m_amp,
#   (2) angles gamma in the transverse plane m_gamma,
#   (3) angles phi in the magnetization plane m_phi,
#   (4) the materials (unit cells) with UNIFORM parameters that shall be investigated
#   (5) some structure that defines the configuration of different grains for magnetization computation
#   (D),(E) electron (phonon)- temperatures Te(p) (at each time step 1 of these 1d arrays is needed)


# (1,2,3,4,5) Let's define a dummy function to construct a sample consisting of different materials (this does not mean atoms but rather a definition for grainsize of at least a unit cell!)
# For this, I construct a class that holds relevant information for the quantum LLB:
class material():
    def __init__(self, name, S, Tc, lamda, muat, kappa_anis, anis_axis, K_0, A_0, Ms, Delta):
        self.name=name                                      # name of the material used for the string representation of the class
        self.S=S                                            # effective spin
        self.Tc=Tc                                          # Curie temperature
        self.J=3*self.S/(self.S+1)*sp.k*self.Tc             # mean field exchange coupling constant
        self.mean_mag_map=self.create_mean_mag_map()         # creates the mean magnetization map over temperature as an interpolation function
        self.lamda=lamda                                    # intrinsic coupling to bath parameter
        self.muat=muat                                      # atomic magnetic moment
        self.kappa_anis=kappa_anis                          # exponent for the temperature dependence of uniaxial anisotropy
        self.anis_axis=anis_axis                            # uniaxials anisotropy axis (x:0, y:1, z:2) other anisotropies are not yet implemented
        self.K_0=K_0                                        # value for the anisotropy at T=0 K in units of J
        self.A_0=A_0                                        # value for the exchange stiffness at T=0 K in units of J*m^2
        self.Ms=Ms                                          #value for the saturation magnetization at T=0K in J/m^3/T
        self.Delta=Delta                                    # in-depth length of the magnetic grains

    def __str__(self):
        return self.name

    # From only these few parameters we can already abstract the mean field magnetization map, which is the basis of all temperature dependent parameters in the quantum LLB
    def create_mean_mag_map(self):
        # This function computes the mean field mean magnetization map by solving the self-consistent equation m=B(m, T)
        # As an output we get an interpolation function of the mean field magnetization at any temperature T<=T_c (this can of course be extended to T>T_c with zeros).
        # I have not worried about m<0 yet but this should be a quick implementation by mirroring the interpolation function later on in the code.

        # Start by defining a unity function m=m:
        def mag(m):
            return m

        # Define the Brillouin function as a function of scalars, as fsolve takes functions of scalars:
        def Brillouin(m, T):
            # This function takes input parameters
            #   (i) magnetization amplitude m_amp_grid (scalar)
            #   (ii) (electron) temperature (scalar)
            # As an output we get the Brillouin function evaluated at (i), (ii) (scalar)

            eta = self.J * m / sp.k / T /self.Tc
            c1 = (2 * self.S + 1) / (2 * self.S)
            c2 = 1 / (2 * self.S)
            bri_func = c1 / np.tanh(c1 * eta) - c2 / np.tanh(c2 * eta)
            return bri_func

        # Then we also need a temperature grid. I'll make it course grained for low temperatures (<0.8*Tc) (small slope) and fine grained for large temperatures (large slope):
        temp_grid=np.array(list(np.arange(0, 0.8, 1e-3))+list(np.arange(0.8, 1+1e-5, 1e-5)))

        # I will define the list of m_eq(T) here and append the solutions of m=B(m, T). It will have the length len(temp_grid) at the end.
        meq_list=[1.]

        # Define a function to find the intersection of m and B(m, T) for given T with scipy:
        def find_intersection_sp(m, Bm, m0):
            return op.fsolve(lambda x: m(x) - Bm(x), m0)

        # Find meq for every temperature, starting point for the search being (1-T/Tc)^(1/2), fill the list
        for i,T in enumerate(temp_grid[1:]):
            # Redefine the Brillouin function to set the temperature parameter (I did not find a more elegant solution to this):
            def Brillouin_2(m):
                return Brillouin(m, T)
            # Get meq:
            meq=find_intersection_sp(mag, Brillouin_2, np.sqrt(1-T))
            if meq[0]<0:            # This is a comletely unwarranted fix for values of meq<0 that fsolve produces. It seems to work though, as the interpolated function plotted by plot_mean_mags() seems clean.
                meq[0]*=-1
            # Append it to list me(T)
            meq_list.append(meq[0])
        meq_list[-1]=0              # This fixes slight computational errors to fix m_eq(Tc)=0 (it produces something like m_eq[-1]=1e-7)
        return ip.interp1d(temp_grid, meq_list)

    def dbrillouin_t1(self):
        return 1 / 4 / self.S ** 2

    def dbrillouin_t2(self):
        return (2 * self.S + 1) ** 2 / 4 / self.S ** 2


    def get_mean_mag(self, T, tc_mask):
        # After creating the map, this function can be called to give m_eq at any temperature
        # The function takes a 1d-array of temperatures as an input (temperature map at each timestep) and returns an array with the respective mean field equilibrium temperatures
        meq=np.zeros(len(T))
        meq[tc_mask]=self.mean_mag_map(T[tc_mask])
        return meq

    def alpha_par(self):
        # This funtion computes the longitudinal damping parameter alpha_parallel
        return 2 * self.lamda / (self.S + 1)

    def qs(self):
        # This function computes the first term of the transverse damping parameter alpha_transverse
        qs = 3 * self.Tc / (2 * self.S + 1)
        return qs

    def chi_par_num(self):
        return 1 / sp.k * self.muat*9.274e-24

    def chi_par_denomm1(self):
        return self.J / sp.k

    def anisotropy(self):
        #This takes mean field magnetization (1d-array of length N (number of grains)), magnetization vectors (dimension 3xN), magnetization amplitudes (length N) and easy axis ([0,1,2] corresponding to [x,y,z])
        return -2*self.K_0


def get_sample():
    # This is a dummy function that should definitely be replaced by outputs from your code. It does not take any input parameters as I define everything here.
    # As an output we get
    #   (i) a 1d list of M materials within the sample (materials on the scale of the grainsize of the macrospins)
    #   (ii) a 1d numpy array of the actual sample consisting of stacked layers of the M materials
    #   (iii-v) magnetization amplitudes and angles

    # Define define three dummy materials with different parameters:
    mat_1 = material('Nickel', 0.5, 630., 0.2, 0.393, 3, 2, 0.45e6, 1e-11, 500e3, 1e-9)
    mat_2 = material('Cobalt', 1e6, 1480., 0.1, 0.393, 3, 2, 0.45e6, 1e-11, 1400e3, 1e-9)
    mat_3 = material('Iron', 2., 1024., 0.05, 2.2, 3, 2, 0.45e6, 1e-11, 200e3, 1e-9)
    # FGT = material ('FGT', 0.5, 220., 0.01, 2.2, 3, 2, 0.45e6, 1e-11, 200e-13, 1e-9)
    # FGT2 = material ('FGT2', 2., 220., 0.01, 2.2, 3, 2, 0.45e6, 1e-11, 200e-13, 1e-9)

    materials = [mat_1, mat_2, mat_3]

    Nickel_1 = [mat_1 for _ in range(10)]
    Cobalt = [mat_2 for _ in range(15)]
    Iron = [mat_3 for _ in range(10)]
    Nickel_2 = [mat_1 for _ in range(25)]

    sample = np.array(Nickel_1 + Cobalt + Iron + Nickel_2)

    # The following constructs a list of lists, containing in list[i] a list of indices of material i in the sample_structure. This will help compute the mean field magnetization only once for every material at each timestep.
    material_grain_indices = []
    for mat in materials:
        material_grain_indices.append([i for i in range(len(sample)) if sample[i] == mat])
    material_grain_indices_flat = [index for mat_list in material_grain_indices for index in mat_list]
    sample_sorter = np.array([material_grain_indices_flat.index(i) for i in np.arange(len(sample))])

    # The following list locates which material is positioned at which grain of the sample. THis will later be used to define an array of material paramters for the whole sample
    mat_locator = [materials.index(grain) for grain in sample]

    # Define initial magnetization on the whole sample (for simplicity uniform) and fully magnetized along the z-axis
    m_amp = np.ones(60)
    m_phi = np.zeros(60)
    m_gamma = np.zeros(60)
    return materials, sample, m_amp, m_phi, m_gamma, material_grain_indices, sample_sorter, mat_locator


# Now I define a function to get the magnetization vector from amplitude and angles:
def get_mag(polar_dat):
    # This function takes as input parameters the amplitude and angles (A, gamma, phi) and puts out a numpy array of dimension 3xlen(sample)
    # with 3 magnetization components for len(sample) grains
    amp = polar_dat[0, :]
    gamma = polar_dat[1, :]
    phi = polar_dat[2, :]
    sin_phi = np.sin(phi)

    mx = amp * sin_phi * np.cos(gamma)
    my = amp * sin_phi * np.sin(gamma)
    mz = amp * np.cos(phi)

    return np.array([mx, my, mz]).T

# The following two function just plot mean field magnetization and its derivative. This can be implemented as something like magnetization.mmag.visualize() once we implement this in the magnetization class I suppose.
def plot_mean_mags(materials):
    #define a temperature grid:
    temps=np.arange(0,2+1e-4, 1e-4)
    tc_mask=temps<1.
    temps[-1]=1.
    for i,m in enumerate(materials):
        mmag=m.get_mean_mag(temps, tc_mask)
        label=str(m)
        plt.plot(temps*m.Tc, mmag, label=label)

    plt.xlabel(r'Temperature [K]', fontsize=16)
    plt.ylabel(r'$m_{\rm{eq}}$', fontsize=16)
    plt.legend(fontsize=14)
    plt.title(r'$m_{\rm{eq}}$ for all materials in sample', fontsize=18)
    plt.savefig('plots/meqtest.pdf')
    plt.show()


# All the above defines the needed material parameters. Now we deal with the actual sample structure. In principle, the interactions
# that will later be computed dynamically get scalar (z.B. exchange interaction) or vectorial outcomes that we can map onto the the material parameters
# meq(T) for each grain in the sample.
# Preperation of material parameters on the sample structure:

# From the sample and the materials within, one can set up an array of exchange-coupling constants between all grains:
def get_exch_coup_sample(materials, sample, mat_loc):
    # This function takes as input parameters:
    #   (i) the 1d-list of magnetic unique materials in the sample (size M)
    #   (ii) the 1d numpy array of the sample, consisting of a material (from class material) at each grain (size N)
    # As an output we get a 2d numpy array of dimension Nx2 for coupling each site with its 2 neighbours in the linear chain of grains.

    # Define a matrix J_mat of dimension len(materials)xlen(materials) with the exchange coupling constants of mat_i and mat_j at J_mat[i][j]=J_mat[j][i]
    J_mat = np.zeros((len(materials), len(materials)))
    # fill the diagonal with the mean field exchange constant of each material:
    for i, mat in enumerate(materials):
        J_mat[i][i] = mat.J
    # define the off-diagonals, namely some values for exchange coupling constants of different materials:
    J_mat[0][1] = 1e-20
    J_mat[1][2] = 5e-20
    J_mat[0][2] = 1e-19
    # symmetrize the matrix so that also elements [i][j] with i>j can be read out:
    for i in range(1, len(materials)):
        for j in range(i):
            J_mat[i][j] = J_mat[j][i]

    # Now we can assign the coupling of each grain to its nearest neighbours by filling the output array with the respective matrix entry:
    # Let's define the output array:
    ex_coup_arr = np.zeros((len(sample), 2))

    # This list can assign the proper matrix elements to the output matrix
    for i, grain in enumerate(sample):
        if i > 0:
            ex_coup_arr[i][0] = J_mat[mat_loc[i]][mat_loc[i - 1]]
        if i < len(sample) - 1:
            ex_coup_arr[i][1] = J_mat[mat_loc[i]][mat_loc[i + 1]]
    return ex_coup_arr


def get_ex_stiff_sample(materials, sample, mat_loc, Ms_sam, Delta2_sam):
    # This computes a grid for the exchange stiffness in analogous fashion to get_exch_coup_sam()
    A_mat = np.zeros((len(materials), len(materials)))
    for i, mat in enumerate(materials):
        A_mat[i][i] = mat.A_0

    A_mat[0][1]=1e-11
    A_mat[1][2]=5e-11
    A_mat[0][2]=2.5e-11

    for i in range(1, len(materials)):
        for j in range(i):
            A_mat[i][j] = A_mat[j][i]

    ex_stiff_arr = np.ones((len(sample), 2)) * A_mat[0][0]

    for i, grain in enumerate(sample):
        if i > 0:
            ex_stiff_arr[i][0] = A_mat[mat_loc[i]][mat_loc[i - 1]]
        if i < len(sample) - 1:
            ex_stiff_arr[i][1] = A_mat[mat_loc[i]][mat_loc[i + 1]]
    return np.divide(ex_stiff_arr, np.multiply(Ms_sam, Delta2_sam)[:, np.newaxis])

def S_sample(sample):
    return np.array([mat.S for mat in sample])

def Tc_sample(sample):
    return np.array([mat.Tc for mat in sample])

def J_sample(sample):
    return np.array([mat.J for mat in sample])

def lamda_sample(sample):
    return np.array([mat.lamda for mat in sample])

def Ms_sample(sample):
    return np.array([mat.Ms for mat in sample])

def Delta2_sample(sample):
    return np.array([mat.Delta**2 for mat in sample])

def muat_sample(sample):
    return np.array([mat.muat for mat in sample])

def get_ani_sample(sample, Ms_sam):
    ani_sam=np.divide(np.array([mat.anisotropy() for mat in sample]), Ms_sam)
    kappa_ani_sam=np.array([mat.kappa_anis for mat in sample])
    ani_perp_sam= np.ones((len(sample), 3))
    for i,mat in enumerate(sample):
        ani_perp_sam[i, mat.anis_axis]=0
    return ani_sam, kappa_ani_sam, ani_perp_sam

def alpha_par_sample(sample):
    return np.array([mat.alpha_par() for mat in sample])

def qs_sample(sample):
    return np.array([mat.qs() for mat in sample])

def dbrillouin_t1_sample(sample):
    return np.array([mat.dbrillouin_t1() for mat in sample])

def dbrillouin_t2_sample(sample):
    return np.array([mat.dbrillouin_t2() for mat in sample])

def chi_par_num_sample(sample):
    return np.array([mat.chi_par_num() for mat in sample])

def chi_par_denomm1_sample(sample):
    return np.array([mat.chi_par_denomm1() for mat in sample])

### Now we deal with the temperature dependence of the parameters. All the following functions are called at every timestep of the dynamical simulation:


def split_sample_T(T, tc_mask, mat_gr_ind, materials):
    T_sep_red=[np.array([T[i] for i in mat_ind])/materials[j].Tc for j, mat_ind in enumerate(mat_gr_ind)]
    tc_mask_sep=[np.array([tc_mask[i] for i in mat_ind]) for mat_ind in mat_gr_ind]
    return T_sep_red, tc_mask_sep


def get_mean_mag_sample_T(mat_gr_ind_flat, materials, T_sep_red, tc_mask_sep):
    Tc_vals = np.array([mat.Tc for mat in materials])
    tc_mask_sep_norm = [np.array(tc_mask) for tc_mask in tc_mask_sep]
    mean_mags = np.array([mat.get_mean_mag(T, tc_mask) for mat, T, tc_mask in zip(materials, T_sep_red, tc_mask_sep_norm)])
    mmag_sam_T_flat = np.concatenate(mean_mags)[mat_gr_ind_flat]
    return mmag_sam_T_flat

def ani_sample_T(mmag_sam_T, K0_sam, kappa_ani_sam):
    return np.multiply(K0_sam,np.power(mmag_sam_T,kappa_ani_sam-2))

def ex_stiff_sample_T(mmag_sam_T, ex_stiff_sam):
    return np.multiply(np.power(mmag_sam_T[:, np.newaxis],2-2),ex_stiff_sam)

def qs_sample_T(qs_sam, mmag_sam_T, T):
    return np.divide(np.multiply(qs_sam,mmag_sam_T),T)

def alpha_par_sample_T(mmag_sam_T, T, alpha_par_sam, qs_sam_T, Tc_sam, under_tc, over_tc, lambda_sam):
    apsT=np.zeros(len(T))
    apsT[under_tc]=alpha_par_sam[under_tc]/np.sinh(2*qs_sam_T[under_tc])
    apsT[over_tc]=lambda_sam[over_tc]*2/3*np.divide(T[over_tc], Tc_sam[over_tc])
    return apsT

def alpha_trans_sample_T(mmag_sam_T, lamda_sam, T, qs_sam_T, Tc_sam, under_tc, over_tc, lambda_sam):
    atsT=np.zeros(len(T))
    atsT[under_tc]=np.multiply(lambda_sam[under_tc], (np.divide(np.tanh(qs_sam_T[under_tc]), qs_sam_T[under_tc])-np.divide(T[under_tc],3*Tc_sam[under_tc])))
    atsT[over_tc]=lambda_sam[over_tc]*2/3*np.divide(T[over_tc], Tc_sam[over_tc])
    return atsT

def eta_sample_T(mmag_sam_T, J_sam, T):
    return J_sam*mmag_sam_T/sp.k/T

def dbrillouin_sample_T(eta_sam_T, S_sam, dbrillouin_t1_sam, dbrillouin_t2_sam):
    two_S_sam=2*S_sam
    x1=np.divide(eta_sam_T,two_S_sam)
    x2=np.divide(np.multiply(eta_sam_T,(two_S_sam+1)),two_S_sam)
    sinh_func=1/np.sinh(np.array([x1,x2]))**2
    dbrillouin_sam_T=dbrillouin_t1_sam*sinh_func[0]-dbrillouin_t2_sam*sinh_func[1]
    return dbrillouin_sam_T

def chi_par_sample_T(chi_par_num_sam, chi_par_denomm1_sam, dbrillouin_sam_T, T, under_tc, over_tc, muat_sam, Tc_sam, J_sam):
    cpsT=np.zeros(len(T))
    cpsT[under_tc]=np.multiply(chi_par_num_sam[under_tc], np.divide(dbrillouin_sam_T, T[under_tc]-np.multiply(chi_par_denomm1_sam[under_tc], dbrillouin_sam_T)))
    cpsT[over_tc]=np.divide(np.multiply(muat_sam[over_tc]*9.274e-24, Tc_sam[over_tc]), J_sam[over_tc]*(T[over_tc]-Tc_sam[over_tc]+1e-1))
    return cpsT



def anis_field(anis_sam_T, m, ani_perp_sam):
    return anis_sam_T[:, np.newaxis]*(m*ani_perp_sam)

def ex_field(ex_stiff_sam_T, m_diff_up, m_diff_down):
    return ex_stiff_sam_T[:,0][:,np.newaxis]*m_diff_up+ex_stiff_sam_T[:,1][:,np.newaxis]*m_diff_down

def th_field(m, m_squared, mmag_sam_T, T, Tc_sam, chi_par_sam_T, under_tc, over_tc):
    factor = 1/chi_par_sam_T
    H_th = np.zeros(len(T))
    H_th[under_tc] = (1-m_squared[under_tc]/mmag_sam_T[under_tc]**2)*factor[under_tc]/2
    H_th[over_tc] = -(1+3/5*Tc_sam[over_tc]/(T[over_tc]-Tc_sam[over_tc]+1e-1))*m_squared[over_tc]*factor[over_tc]
    return H_th[:, np.newaxis]*m

def mag_incr(materials, sample, m_amp, m_phi, m_gamma, mat_gr_ind, mat_gr_ind_flat, mat_loc, Ms_sam, ex_stiff_sam,
             S_sam, Tc_sam, J_sam, lamda_sam, muat_sam, K0_sam, kappa_ani_sam, ani_perp_sam, alpha_par_sam, qs_sam,
             dbrillouin_t1_sam, dbrillouin_t2_sam, chi_par_num_sam, chi_par_denomm1_sam, m, Te, H_ext, gamma, dt):
    # some operations of the magnetization we need:
    m_squared = np.sum(np.power(m, 2), axis=-1)
    m_diff_down = -np.concatenate((np.diff(m, axis=0), np.zeros((1, 3))), axis=0)
    m_diff_up = np.roll(m_diff_down, 1)

    # at every timestep, we have to calculate the temperature dependent parameters, so let's call all the functions defined before
    t_reduced = np.divide(Te, Tc_sam)
    under_tc = t_reduced < 1.
    over_tc = ~under_tc
    Temp_sep, tc_mask_sep = split_sample_T(Te, under_tc, mat_gr_ind, materials)
    mmag_sam_T = get_mean_mag_sample_T(mat_gr_ind_flat, materials, Temp_sep, tc_mask_sep)
    anis_sam_T = ani_sample_T(mmag_sam_T, K0_sam, kappa_ani_sam)
    ex_stiff_sam_T = ex_stiff_sample_T(mmag_sam_T, ex_stiff_sam)
    qs_sam_T = qs_sample_T(qs_sam, mmag_sam_T, Te)
    eta_sam_T = eta_sample_T(mmag_sam_T, J_sam, Te)
    dbrillouin_sam_T = dbrillouin_sample_T(eta_sam_T[under_tc], S_sam[under_tc], dbrillouin_t1_sam[under_tc],
                                           dbrillouin_t2_sam[under_tc])
    chi_par_sam_T = chi_par_sample_T(chi_par_num_sam, chi_par_denomm1_sam, dbrillouin_sam_T, Te, under_tc, over_tc,
                                     muat_sam, Tc_sam, J_sam)

    # from all these we can define the effective field
    H_ex = ex_field(ex_stiff_sam_T, m_diff_up, m_diff_down)
    H_ani = anis_field(anis_sam_T, m, ani_perp_sam)
    H_th = th_field(m, m_squared, mmag_sam_T, Te, Tc_sam, chi_par_sam_T, under_tc, over_tc)

    H_eff = H_ani + H_ex + H_th + H_ext

    # and the damping parameters
    alpha_par_sam_T = alpha_par_sample_T(mmag_sam_T, Te, alpha_par_sam, qs_sam_T, Tc_sam, under_tc, over_tc, lamda_sam)
    alpha_trans_sam_T = alpha_trans_sample_T(mmag_sam_T, lamda_sam, Te, qs_sam_T, Tc_sam, under_tc, over_tc, lamda_sam)

    # we will precompute the prefactors that are not of dimension len(sample)x3:

    pref_trans = np.divide(alpha_trans_sam_T, m_squared)
    pref_long = np.multiply(np.divide(alpha_par_sam_T, m_squared), np.einsum('ij,ij->i', m, H_eff))

    # and all the cross products:
    m_rot = np.cross(m, H_eff)  # precessional term
    m_trans = np.cross(m, m_rot)  # transverse damping term

    trans_damp = np.multiply(pref_trans[:, np.newaxis], m_trans)
    long_damp = np.multiply(pref_long[:, np.newaxis], m)

    # Now compute the magnetization increment
    dm = gamma * dt * (-m_rot - trans_damp + long_damp)

    return dm


def run_LLB(tes, dt):
    starttime = time.time()
    # let's define some constants for simulation:
    gamma = 1.76e11  # gyromagnetic ratio in (Ts)^{-1}
    H_ext = np.array([[0, 0, 0] for _ in range(60)])  # external field in T

    # load a sample and call the functions to get all parameters on the sample structure:
    materials, sample, m_amp, m_phi, m_gamma, mat_gr_ind, mat_gr_ind_flat, mat_loc = get_sample()

    Ms_sam = Ms_sample(sample)
    # exch_coup_const_sam=get_exch_coup_sample(materials, sample, mat_loc)
    Delta2_sam = Delta2_sample(sample)
    S_sam = S_sample(sample)
    Tc_sam = Tc_sample(sample)
    J_sam = J_sample(sample)
    lamda_sam = lamda_sample(sample)
    muat_sam = muat_sample(sample)
    ex_stiff_sam = get_ex_stiff_sample(materials, sample, mat_loc, Ms_sam, Delta2_sam)
    K0_sam, kappa_ani_sam, ani_perp_sam = get_ani_sample(sample, Ms_sam)
    alpha_par_sam = alpha_par_sample(sample)
    qs_sam = qs_sample(sample)
    dbrillouin_t1_sam = dbrillouin_t1_sample(sample)
    dbrillouin_t2_sam = dbrillouin_t2_sample(sample)
    chi_par_num_sam = chi_par_num_sample(sample)
    chi_par_denomm1_sam = chi_par_denomm1_sample(sample)

    # initialize the starting magnetization
    m=get_mag(np.array([m_amp, m_phi, m_gamma]))
    mag_map = [m]

    initime = time.time()
    print('Magnetization parameters initialized. Time spent:', str(initime - starttime), 's')

    for i, Te in enumerate(tes):
        dm1 = mag_incr(materials, sample, m_amp, m_phi, m_gamma, mat_gr_ind, mat_gr_ind_flat, mat_loc, Ms_sam,
                       ex_stiff_sam, S_sam, Tc_sam, J_sam, lamda_sam, muat_sam, K0_sam, kappa_ani_sam, ani_perp_sam,
                       alpha_par_sam, qs_sam, dbrillouin_t1_sam, dbrillouin_t2_sam, chi_par_num_sam,
                       chi_par_denomm1_sam, m, Te, H_ext, gamma, dt)
        m_heun = m + dm1
        dm2 = mag_incr(materials, sample, m_amp, m_phi, m_gamma, mat_gr_ind, mat_gr_ind_flat, mat_loc, Ms_sam,
                       ex_stiff_sam, S_sam, Tc_sam, J_sam, lamda_sam, muat_sam, K0_sam, kappa_ani_sam, ani_perp_sam,
                       alpha_par_sam, qs_sam, dbrillouin_t1_sam, dbrillouin_t2_sam, chi_par_num_sam,
                       chi_par_denomm1_sam, m_heun, Te, H_ext, gamma, dt)
        newmag = m + np.divide((dm1 + dm2), 2)
        m = newmag
        mag_map.append(m)
    endtime = time.time()
    print('Magnetization map created. Time spent on dynamical part:' , str(endtime - initime) , 's')
    print('Total time spent on magnetization dynamics:' , str(endtime-starttime) , 's')
    return np.array(mag_map)

delay=np.load('temp_test/delays.npy')
teNi1=np.load('temp_test/tesNickel0.npy')
teCo2=np.load('temp_test/tesCobalt1.npy')
teFe3=np.load('temp_test/tesIron2.npy')
teNi4=np.load('temp_test/tesNickel3.npy')
tes=np.append(teNi1, teCo2, axis=1)
tes=np.append(tes, teFe3, axis=1)
tes=np.append(tes, teNi4, axis=1)
#tes=np.array([[tes[i,0]] for i in range(len(delay))])*0.7

# paul_dat=open('temp_test/paul_data.txt', 'r').readlines()
# sep_dat=[line.split() for line in paul_dat]
# float_dat=np.array([[float(num) for num in line] for line in sep_dat])
#
# delay=np.array(float_dat[:,0])[:51]
# new_delay=np.arange(0,delay[-1], 1e-4)#[:10001]

# tes=float_dat[:,5][:51]
# new_tes=np.array(ip.interp1d(delay, tes)(new_delay))#[:10001]
# new_tes = np.reshape(new_tes,(-1,1))
#
# mxs= float_dat[:,1]
# mys=float_dat[:,2]
# mzs=float_dat[:,3]
# m2s=float_dat[:,4]


mag_map=run_LLB(tes, 1e-16)

color_data = mag_map[:, :, 2].T
plt.imshow(color_data, cmap='hot', aspect='auto')
plt.ylabel(r'grain position')
plt.xlabel(r'time delay [ps]')
plt.colorbar()
plt.show()
