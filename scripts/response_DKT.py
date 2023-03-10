###=============================================================###
###  Compute second-order response function using DKT			###
###=============================================================###

import sys
import numpy as np
from matplotlib import pyplot as plt
plot_freq=True
plot_time=True
lmax_all=100
lmax_all=None
lmax_all=15
f_target = 1.0
print("freq target {}".format(f_target))
###================================================================
### read some parameters from command line

### usage
if len(sys.argv) != 2 and len(sys.argv) != 4:
    sys.exit(
        "Usage: python {} beta   or \n python {} beta lambda norm".format(sys.argv[0])
    )

### beta: inverse temperature (au)
beta = float(sys.argv[1])
print("beta = ", beta)

###================================================================

###=====================================================================
### import DKT data
### Note: order is [time,time,DKT.real,DKT.imag]

filename = "DKT.dat"
temp_t, temp_r, temp_i = np.loadtxt(filename, unpack=True, usecols=[1, 2, 3])
DKT = temp_r + 1.0j * temp_i

### define time variables

ndim = int(np.power(len(temp_t), 0.5))
print("ndim = ", ndim)

time = np.zeros(ndim)
for i in range(ndim):
    time[i] = temp_t[i]

dt = time[1] - time[0]
print("dt=", dt)

###================================================================

###=====================================================================
### import standard functions
### Note: order is [time,time,C.real,C.imag]

# C_ABC
filename = "C_ABC.dat"
temp_r, temp_i = np.loadtxt(filename, unpack=True, usecols=[2, 3])
C_ABC = temp_r + 1.0j * temp_i

# C_ACB
filename = "C_ACB.dat"
temp_r, temp_i = np.loadtxt(filename, unpack=True, usecols=[2, 3])
C_ACB = temp_r + 1.0j * temp_i

# C_CBA
filename = "C_CBA.dat"
temp_r, temp_i = np.loadtxt(filename, unpack=True, usecols=[2, 3])
C_CBA = temp_r + 1.0j * temp_i

# C_BCA
filename = "C_BCA.dat"
temp_r, temp_i = np.loadtxt(filename, unpack=True, usecols=[2, 3])
C_BCA = temp_r + 1.0j * temp_i

###================================================================
# KT_PB
filename = "KT_PB_ABC.dat"
temp_r, temp_i = np.loadtxt(filename, unpack=True, usecols=[2, 3])
KTPB_ABC = (temp_r + 1.0j * temp_i)
# KT_PBp
filename = "KT_PB_ABCp.dat"
temp_r, temp_i = np.loadtxt(filename, unpack=True, usecols=[2, 3])
KTPBp_ABC = (temp_r + 1.0j * temp_i)

# KT_PB
filename = "KT_PB_CBA.dat"
temp_r, temp_i = np.loadtxt(filename, unpack=True, usecols=[2, 3])
KTPB_CBA = (temp_r + 1.0j * temp_i)
# KT_PBp
filename = "KT_PB_CBAp.dat"
temp_r, temp_i = np.loadtxt(filename, unpack=True, usecols=[2, 3])
KTPBp_CBA = (temp_r + 1.0j * temp_i)


###================================================================
# DKTp
filename = "DKTp1.dat"
temp_r, temp_i = np.loadtxt(filename, unpack=True, usecols=[2, 3])
DKTp1 = (temp_r + 1.0j * temp_i)

# DKTp
filename = "DKTp2.dat"
temp_r, temp_i = np.loadtxt(filename, unpack=True, usecols=[2, 3])
DKTp2 = (temp_r + 1.0j * temp_i)

# DKTpp
filename = "DKTpp.dat"
temp_r, temp_i = np.loadtxt(filename, unpack=True, usecols=[2, 3])
DKTpp = (temp_r + 1.0j * temp_i)

# DKTpp2
#filename = "DKTpp2.dat"
#temp_r, temp_i = np.loadtxt(filename, unpack=True, usecols=[2, 3])
#DKTpp2 = (temp_r + 1.0j * temp_i)
DKTpp2=np.zeros(DKTpp.shape)
###================================================================
### reshape data into 2D array format

DKT = np.reshape(DKT, [ndim, ndim])
C_ABC = np.reshape(C_ABC, [ndim, ndim])
C_CBA = np.reshape(C_CBA, [ndim, ndim])
C_ACB = np.reshape(C_ACB, [ndim, ndim])
C_BCA = np.reshape(C_BCA, [ndim, ndim])

KTPB_ABC = np.reshape(KTPB_ABC, [ndim, ndim])
KTPBp_ABC = np.reshape(KTPBp_ABC, [ndim, ndim])
KTPB_CBA = np.reshape(KTPB_CBA, [ndim, ndim])
KTPBp_CBA = np.reshape(KTPBp_CBA, [ndim, ndim])

DKTp1 = np.reshape(DKTp1, [ndim, ndim])
DKTp2 = np.reshape(DKTp2, [ndim, ndim])
DKTpp = np.reshape(DKTpp, [ndim, ndim])
DKTpp2 = np.reshape(DKTpp2, [ndim, ndim])

###================================================================

###================================================================
### Damp function

n_order = 2

# tau = 0.5*time[ndim-1]
tau = 13.0
tau = 10.0
print("tau = ", tau)
   

print(tau)
print(time.shape)

for i in range(ndim):
    for j in range(ndim):
        delta = np.power(np.abs(time[i]) / tau, n_order) + np.power(
            np.abs(time[j]) / tau, n_order
        )
        DKT[i, j] *= np.exp(-delta)
        C_ABC[i, j] *= np.exp(-delta)
        C_CBA[i, j] *= np.exp(-delta)
        C_ACB[i, j] *= np.exp(-delta)
        C_BCA[i, j] *= np.exp(-delta)

        KTPB_ABC[i, j] *= np.exp(-delta)
        KTPBp_ABC[i, j] *= np.exp(-delta)
        KTPB_CBA[i, j] *= np.exp(-delta)
        KTPBp_CBA[i, j] *= np.exp(-delta)

        DKTp1[i, j] *= np.exp(-delta)
        DKTp2[i, j] *= np.exp(-delta)
        DKTpp[i, j] *= np.exp(-delta)
        DKTpp2[i, j] *= np.exp(-delta)

###================================================================

###================================================================
### get symm and asym DKT
### Note: The symm and asym are computed as:
###       > DKT^{symm} = 2.*Re{DKT}
###       > DKT^{asym} = 2.*Im{DKT}
### Note that both correlations are taken to be purely 'real', whereas the asym
### should be purely imaginary. This is corrected when computing the FT (see below).

DKT_symm = 2.0 * DKT.real
DKT_asym = 2.0 * DKT.imag

KTPB_ABC = KTPB_ABC
KTPBp_ABC = KTPBp_ABC
KTPB_CBA = KTPB_CBA
KTPBp_CBA = KTPBp_CBA

DKTp1 = DKTp1
DKTp2 = DKTp2
DKTpp = 2.0*DKTpp.real 
DKTpp2 = 2.0*DKTpp2.real 

###================================================================
### Perform 2D FFT
### Note: To avoid phase shifts, we shift the data into the format pos-neg first

DKT_fft = np.fft.fft2(np.fft.fftshift(DKT))
DKT_symm_fft = np.fft.fft2(np.fft.fftshift(DKT_symm))
DKT_asym_fft = np.fft.fft2(np.fft.fftshift(DKT_asym))


C_ABC_fft = np.fft.fft2(np.fft.fftshift(C_ABC))
C_CBA_fft = np.fft.fft2(np.fft.fftshift(C_CBA))
C_ACB_fft = np.fft.fft2(np.fft.fftshift(C_ACB))
C_BCA_fft = np.fft.fft2(np.fft.fftshift(C_BCA))

KTPB_ABC_fft = np.fft.fft2(np.fft.fftshift(KTPB_ABC))
KTPBp_ABC_fft = np.fft.fft2(np.fft.fftshift(KTPBp_ABC))
KTPB_CBA_fft = np.fft.fft2(np.fft.fftshift(KTPB_CBA))
KTPBp_CBA_fft = np.fft.fft2(np.fft.fftshift(KTPBp_CBA))

DKTp1_fft = np.fft.fft2(np.fft.fftshift(DKTp1))
DKTp2_fft = np.fft.fft2(np.fft.fftshift(DKTp2))
DKTpp_fft = np.fft.fft2(np.fft.fftshift(DKTpp))
DKTpp2_fft = np.fft.fft2(np.fft.fftshift(DKTpp2))

###================================================================

###================================================================
### Normalize FFT

DKT_fft *= dt * dt
DKT_symm_fft *= dt * dt
# multiply by -1 due to missing imaginary part in definition of asym
DKT_asym_fft *= (    -1.0 * dt * dt)

C_ABC_fft *= dt * dt
C_CBA_fft *= dt * dt
C_ACB_fft *= dt * dt
C_BCA_fft *= dt * dt

KTPB_ABC_fft *= dt * dt
KTPBp_ABC_fft *= dt * dt
KTPB_CBA_fft *= dt * dt
KTPBp_CBA_fft *= dt * dt

DKTp1_fft *= dt * dt
DKTp2_fft *= dt * dt
DKTpp_fft *= dt * dt
DKTpp2_fft *= dt * dt

###================================================================

###================================================================
### Get frequencies of FFT

freq = np.fft.fftfreq(ndim, dt)
freq *= 2.0 * np.pi

###================================================================

###================================================================
### Shift FFT to the format neg-pos

DKT_fft = np.fft.fftshift(DKT_fft)
DKT_symm_fft = np.fft.fftshift(DKT_symm_fft)
DKT_asym_fft = np.fft.fftshift(DKT_asym_fft)

C_ABC_fft = np.fft.fftshift(C_ABC_fft)
C_CBA_fft = np.fft.fftshift(C_CBA_fft)
C_ACB_fft = np.fft.fftshift(C_ACB_fft)
C_BCA_fft = np.fft.fftshift(C_BCA_fft)

KTPB_ABC_fft = np.fft.fftshift(KTPB_ABC_fft)
KTPBp_ABC_fft = np.fft.fftshift(KTPBp_ABC_fft)
KTPB_CBA_fft = np.fft.fftshift(KTPB_CBA_fft)
KTPBp_CBA_fft = np.fft.fftshift(KTPBp_CBA_fft)
DKTp1_fft = np.fft.fftshift(DKTp1_fft)
DKTp2_fft = np.fft.fftshift(DKTp2_fft)
DKTpp_fft = np.fft.fftshift(DKTpp_fft)
DKTpp2_fft = np.fft.fftshift(DKTpp2_fft)

freq = np.fft.fftshift(freq)

###================================================================

###================================================================
### Computing the Q+ and Q- terms
### Note: Different limiting cases of w1/w2 are considered individually.

Q1 = np.zeros([ndim, ndim])
Q2 = np.zeros([ndim, ndim])
Qp = np.zeros([ndim, ndim])
Qm = np.zeros([ndim, ndim])

for i in range(ndim):
    for j in range(ndim):

        beta_i = beta * freq[i]
        beta_j = beta * freq[j]
        beta_bar = beta_i + beta_j

        if freq[i] == 0.0:
            if freq[j] == 0.0:
                Q1[i, j] = 0.0
                Q2[i, j] = 0.0
            else:
                Q1[i, j] = (
                    -(1.0 - np.exp(-beta_j))
                    * (beta_j**2)
                    / (-beta_j * np.exp(-beta_j) - np.exp(-beta_j) + 1.0)
                )
                Q2[i, j] = (
                    -(1.0 - np.exp(+beta_j))
                    * (beta_j**2)
                    / (+beta_j * np.exp(+beta_j) - np.exp(+beta_j) + 1.0)
                )

        elif freq[j] == 0.0:
            if freq[i] == 0.0:
                Q1[i, j] = 0.0
                Q2[i, j] = 0.0
            else:
                Q1[i, j] = (
                    -(1.0 - np.exp(-beta_i))
                    * (beta_i**2)
                    / (np.exp(-beta_i) + beta_i - 1.0)
                )
                Q2[i, j] = (
                    -(1.0 - np.exp(+beta_i))
                    * (beta_i**2)
                    / (np.exp(+beta_i) - beta_i - 1.0)
                )

        elif freq[j] == -freq[i]:
            Q1[i, j] = 0.0
            Q2[i, j] = 0.0
        else:
            Q1[i, j] = (
                -(np.exp(+beta_j) - np.exp(-beta_i))
                * beta_i
                * beta_j
                * beta_bar
                / (np.exp(-beta_i) * beta_j + np.exp(+beta_j) * beta_i - beta_bar)
            )
            Q2[i, j] = (
                -(np.exp(-beta_j) - np.exp(+beta_i))
                * beta_i
                * beta_j
                * beta_bar
                / (np.exp(+beta_i) * beta_j + np.exp(-beta_j) * beta_i - beta_bar)
            )

        Qp[i, j] = 0.5 * (Q1[i, j] + Q2[i, j])
        Qm[i, j] = 0.5 * (Q1[i, j] - Q2[i, j])
        # Qp[i,j] = 1.0
        # Qm[i,j] = 1.0
###================================================================

###================================================================
### Computing terms of the response function

Resp_symm = np.zeros([ndim, ndim])
Resp_asym = np.zeros([ndim, ndim])
Resp_kubo = np.zeros([ndim, ndim])
Resp_std = np.zeros([ndim, ndim])
Resp_new = np.zeros([ndim, ndim])
Resp_new2= np.zeros([ndim, ndim])

for i in range(ndim):
    for j in range(ndim):

        Resp_symm[i, j] = Qp[i, j] * DKT_symm_fft[i, j].real

        Resp_asym[i, j] = Qm[i, j] * DKT_asym_fft[i, j].imag

        Resp_kubo[i, j] = Resp_symm[i, j] + Resp_asym[i, j]

        Resp_std[i, j] = -(
            C_ABC_fft[i, j] + C_CBA_fft[i, j] - C_ACB_fft[i, j] - C_BCA_fft[i, j]
        ).real
        Resp_new[i,j] = (beta**2)* DKTpp_fft[i,j].real - beta * KTPBp_ABC_fft[i,j].real
        Resp_new2[i,j] = beta * KTPBp_CBA_fft[i,j].real
if False:
    try:
        lamb = float(sys.argv[2])
        norm = float(sys.argv[3])
        Resp_symm /= lamb * norm
        Resp_asym /= lamb * norm
        Resp_kubo /= lamb * norm
        Resp_std /= lamb * norm
        print("Lambda {}".format(lamb))
    except:
        pass

###================================================================

###================================================================
### Define functions to plot

plt.rcParams["axes.linewidth"] = 1.5

color = ["black", "white"]

cm_color = "jet"
cm_color = "bwr"

lw = 1.0

labelsz = 12.0

ticksz = 11

nlines = 1

figsize = [8, 8]

titlesz = 10

nlevels = 100  


def plot_2d(x, y, z, figtitle="", title="", xlabel="", ylabel="", axis_x="", axis_y=""):
    """Contour plot"""

    ### define figure
    fig, ax = plt.subplots(figsize=figsize, sharex=True, sharey=True)

    ### define levels
    lmax = np.max(np.abs(z))
    lmin = -lmax
    ldel = (lmax - lmin) / nlevels
    levels = np.arange(lmin, lmax + ldel, ldel)
    idx = np.argmin(np.abs(levels))
    levels = np.delete(levels, idx)  # remove 0.0 contour line

    ### plot data
    CS0 = ax.contourf(x, y, np.transpose(z1), cmap=cm_color, levels=levels)
    CL0 = ax.contour(CS0, colors="k", linewidths=lw)

    ### plot colorbars
    CB0 = fig.colorbar(CS0)

    ### define axis limits
    if axis_x:
        ax.set_xlim(axis_x)
    if axis_y:
        ax.set_ylim(axis_y)

    ### define labels

    fig.suptitle(figtitle, size=titlesz)

    ax.set_title(title1, size=titlesz)

    ax.set_xlabel(xlabel1, size=labelsz)

    ax.set_ylabel(ylabel1, size=labelsz)

    return fig


def plot_2d_2(
    x,
    y,
    z1,
    z2,
    figtitle="",
    title1="",
    xlabel1="",
    ylabel1="",
    title2="",
    xlabel2="",
    ylabel2="",
    axis_x="",
    axis_y="",
):
    """Contour plot [x2]"""

    ### define figure
    fig, ax = plt.subplots(2, figsize=figsize, sharex=True, sharey=True)

    ### define levels
    lmax = max(np.max(np.abs(z1)), np.max(np.abs(z2)))
    if lmax_all is not None:
        lmax = lmax_all
    lmin = -lmax
    ldel = (lmax - lmin) / nlevels
    # print(lmin,lmax,ldel)
    # 	if (np.abs(lmin)<0.00000001):
    # 	    lmin=-569.1726471224425
    # 	    lmax=569.1726471224425
    # 	    ldel=22.7669058848977
    levels = np.arange(lmin, lmax + ldel, ldel)
    idx = np.argmin(np.abs(levels))
    levels = np.delete(levels, idx)  # remove 0.0 contour line

    ### plot data
    CS0 = ax[0].contourf(x, y, np.transpose(z1), cmap=cm_color, levels=levels)
    # AL#	CL0 = ax[0].contour(CS0,colors='k',linewidths=lw)

    CS1 = ax[1].contourf(x, y, np.transpose(z2), cmap=cm_color, levels=levels)
    # AL#	CL1 = ax[1].contour(CS1,colors='k',linewidths=lw)

    ### plot colorbars
    CB0 = fig.colorbar(CS0, ax=ax[0])
    CB1 = fig.colorbar(CS1, ax=ax[1])

    ### define axis limits
    if axis_x:
        ax[0].set_xlim(axis_x)
        ax[1].set_xlim(axis_x)
    if axis_y:
        ax[0].set_ylim(axis_y)
        ax[1].set_ylim(axis_y)

    ### define labels

    fig.suptitle(figtitle, size=titlesz)

    ax[0].set_title(title1, size=titlesz)
    ax[1].set_title(title2, size=titlesz)

    ax[0].set_xlabel(xlabel1, size=labelsz)
    ax[1].set_xlabel(xlabel2, size=labelsz)

    ax[0].set_ylabel(ylabel1, size=labelsz)
    ax[1].set_ylabel(ylabel2, size=labelsz)

    return fig


def plot_2d_3(
    x,
    y,
    z1,
    z2,
    z3,
    figtitle="",
    title1="",
    xlabel1="",
    ylabel1="",
    title2="",
    xlabel2="",
    ylabel2="",
    title3="",
    xlabel3="",
    ylabel3="",
    axis_x="",
    axis_y="",
):
    """Contour plot [x3]"""

    ### define figure
    fig, ax = plt.subplots(3, figsize=figsize, sharex=True, sharey=True)

    ### define levels
    lmax = max(np.max(np.abs(z1)), np.max(np.abs(z2)), np.max(np.abs(z3)))
    lmin = -lmax
    ldel = (lmax - lmin) / nlevels
    levels = np.arange(lmin, lmax + ldel, ldel)
    idx = np.argmin(np.abs(levels))
    levels = np.delete(levels, idx)  # remove 0.0 contour line

    ### plot data
    CS0 = ax[0].contourf(x, y, np.transpose(z1), cmap=cm_color, levels=levels)
    CL0 = ax[0].contour(CS0, colors="k", linewidths=lw)

    CS1 = ax[1].contourf(x, y, np.transpose(z2), cmap=cm_color, levels=levels)
    CL1 = ax[1].contour(CS1, colors="k", linewidths=lw)

    CS2 = ax[2].contourf(x, y, np.transpose(z3), cmap=cm_color, levels=levels)
    CL2 = ax[2].contour(CS2, colors="k", linewidths=lw)

    ### plot colorbars
    CB0 = fig.colorbar(CS0, ax=ax[0])
    CB1 = fig.colorbar(CS1, ax=ax[1])
    CB2 = fig.colorbar(CS2, ax=ax[2])

    ### define axis limits
    if axis_x:
        ax[0].set_xlim(axis_x)
        ax[1].set_xlim(axis_x)
        ax[2].set_xlim(axis_x)
    if axis_y:
        ax[0].set_ylim(axis_y)
        ax[1].set_ylim(axis_y)
        ax[2].set_ylim(axis_y)

    ### define labels

    fig.suptitle(figtitle, size=titlesz)

    ax[0].set_title(title1, size=titlesz)
    ax[1].set_title(title2, size=titlesz)
    ax[2].set_title(title3, size=titlesz)

    ax[0].set_xlabel(xlabel1, size=labelsz)
    ax[1].set_xlabel(xlabel2, size=labelsz)
    ax[2].set_xlabel(xlabel3, size=labelsz)

    ax[0].set_ylabel(ylabel1, size=labelsz)
    ax[1].set_ylabel(ylabel2, size=labelsz)
    ax[2].set_ylabel(ylabel3, size=labelsz)

    return fig


def plot_1d(x, y, figtitle="", title="", xlabel="", ylabel=""):
    """1d plot"""

    ### define figure
    fig, ax = plt.subplots()

    ### plot data
    ax.plot(x, y)

    ### define labels

    fig.suptitle(figtitle, size=titlesz)

    ax.set_title(title, size=titlesz)

    ax.set_xlabel(xlabel, size=labelsz)

    ax.set_ylabel(ylabel, size=labelsz)

    return fig


def plot_1d_list(x, y, figtitle="", title="", xlabel="", ylabel="", legend=""):
    """1d plot where y is a list of curves"""

    ### define figure
    fig, ax = plt.subplots()

    ### define legend
    if not legend:
        legend = np.zeros(len(y))

    ### plot data
    for i in range(len(y)):
        ax.plot(x, y[i], label=legend[i])

    ### define labels

    fig.suptitle(figtitle, size=titlesz)

    ax.set_title(title, size=titlesz)

    ax.set_xlabel(xlabel, size=labelsz)

    ax.set_ylabel(ylabel, size=labelsz)

    ax.legend()

    return fig


###================================================================
###======================== PLOTS and plots =======================
###================================================================

### Plot functions in time-domain

if plot_time:
 if False:
    xlabel = "t1"
    ylabel = "t2"
    axis_x = [-20.0, 20.0]
    axis_y = [-20.0, 20.0]

    ### DKT.real/DKT.imag

    title1 = "DKT.real"
    title2 = "DKT.imag"
    legend = [title1, title2]

    z1 = DKT.real
    z2 = DKT.imag
    fig1 = plot_2d_2(
        time,
        time,
        z1,
        z2,
        title1=title1,
        title2=title2,
        xlabel2=xlabel,
        ylabel1=ylabel,
        ylabel2=ylabel,
        axis_x=axis_x,
        axis_y=axis_y,
    )

 if False: #ALBERTO
    xlabel = "t1"
    ylabel = "t2"
    axis_x = [-20.0, 20.0]
    axis_y = [-20.0, 20.0]

    ### DKT.real/DKT.imag

    title1 = "DKT.real"
    title2 = "DKT.imag"
    legend = [title1, title2]

    z1 = DKT.real
    z2 = DKT.imag
    fig1 = plot_2d_2(
        time,
        time,
        z1,
        z2,
        title1=title1,
        title2=title2,
        xlabel2=xlabel,
        ylabel1=ylabel,
        ylabel2=ylabel,
        axis_x=axis_x,
        axis_y=axis_y,
    )

    if False:
        legend = [title1]
        idx = int(ndim//4)
        data = [DKT[:, idx].real]
        title = "cut at t2={}".format(time[idx])
        fig1a = plot_1d_list(time, data, title=title, xlabel="t", legend=legend)

        legend = [title2]
        data = [DKT[:, idx].imag]
        title = "cut at t2={}".format(time[idx])
        fig1b = plot_1d_list(time, data, title=title, xlabel="t", legend=legend)

        legend = [title1]
        idx = int(ndim//4)
        data = [DKT[idx, :].real]
        title = "cut at t1={}".format(time[idx])
        fig1c = plot_1d_list(time, data, title=title, xlabel="t", legend=legend)

        legend = [title2]
        data = [DKT[idx, :].imag]
        title = "cut at t1={}".format(time[idx])
        fig1d = plot_1d_list(time, data, title=title, xlabel="t", legend=legend)

 if False: #ALBERTO
    xlabel = "t1"
    ylabel = "t2"
    axis_x = [-20.0, 20.0]
    axis_y = [-20.0, 20.0]

    ### DKT.real/DKT.imag

    title1 = "DKTp1.real"
    title2 = "DKTp1.imag"
    legend = [title1, title2]

    z1 = DKTp1.real
    z2 = DKTp1.imag
    fig1 = plot_2d_2(
        time,
        time,
        z1,
        z2,
        title1=title1,
        title2=title2,
        xlabel2=xlabel,
        ylabel1=ylabel,
        ylabel2=ylabel,
        axis_x=axis_x,
        axis_y=axis_y,
    )
 if False: #ALBERTO
    xlabel = "t1"
    ylabel = "t2"
    axis_x = [-20.0, 20.0]
    axis_y = [-20.0, 20.0]

    ### DKT.real/DKT.imag

    title1 = "DKTp2.real"
    title2 = "DKTp2.imag"
    legend = [title1, title2]

    z1 = DKTp2.real
    z2 = DKTp2.imag
    fig1 = plot_2d_2(
        time,
        time,
        z1,
        z2,
        title1=title1,
        title2=title2,
        xlabel2=xlabel,
        ylabel1=ylabel,
        ylabel2=ylabel,
        axis_x=axis_x,
        axis_y=axis_y,
    )

 if True: #ALBERTO
    xlabel = "t1"
    ylabel = "t2"
    axis_x = [-20.0, 20.0]
    axis_y = [-20.0, 20.0]

    ### DKT.real/DKT.imag

    title1 = "DKTpp.real"
    title2 = "DKTpp.imag"
    legend = [title1, title2]

    z1 = DKTpp.real
    z2 = DKTpp.imag
    fig1 = plot_2d_2(
        time,
        time,
        z1,
        z2,
        title1=title1,
        title2=title2,
        xlabel2=xlabel,
        ylabel1=ylabel,
        ylabel2=ylabel,
        axis_x=axis_x,
        axis_y=axis_y,
    )

 if True: #ALBERTO
    xlabel = "t1"
    ylabel = "t2"
    axis_x = [-20.0, 20.0]
    axis_y = [-20.0, 20.0]

    ### DKT.real/DKT.imag

    title1 = "KTPBp_ABC.real"
    title2 = "KTPBp_ABC.imag"
    legend = [title1, title2]

    z1 = KTPBp_ABC.real
    z2 = KTPBp_ABC.imag
    fig1 = plot_2d_2(
        time,
        time,
        z1,
        z2,
        title1=title1,
        title2=title2,
        xlabel2=xlabel,
        ylabel1=ylabel,
        ylabel2=ylabel,
        axis_x=axis_x,
        axis_y=axis_y,
    )

 if False: #ALBERTO
    xlabel = "t1"
    ylabel = "t2"
    axis_x = [-20.0, 20.0]
    axis_y = [-20.0, 20.0]

    ### DKT.real/DKT.imag

    title1 = "KTPBp_ABC.real"
    title2 = "KTPBp_ABC.imag"
    legend = [title1, title2]

    z1 = KTPBp_ABC.real
    z2 = KTPBp_ABC.imag
    fig1 = plot_2d_2(
        time,
        time,
        z1,
        z2,
        title1=title1,
        title2=title2,
        xlabel2=xlabel,
        ylabel1=ylabel,
        ylabel2=ylabel,
        axis_x=axis_x,
        axis_y=axis_y,
    )

 if False: #ALBERTO
    xlabel = "t1"
    ylabel = "t2"
    axis_x = [-20.0, 20.0]
    axis_y = [-20.0, 20.0]

    ### DKT.real/DKT.imag

    title1 = "KTPB_CBA.real"
    title2 = "KTPB_CBA.imag"
    legend = [title1, title2]

    z1 = KTPB_CBA.real
    z2 = KTPB_CBA.imag
    fig1 = plot_2d_2(
        time,
        time,
        z1,
        z2,
        title1=title1,
        title2=title2,
        xlabel2=xlabel,
        ylabel1=ylabel,
        ylabel2=ylabel,
        axis_x=axis_x,
        axis_y=axis_y,
    )
    title1 = "DKT.real"
    title2 = "KTPB_ABC.real"
    legend = [title1, title2]

    z1 = DKT.real
    z2 = KTPB_ABC.real
    fig1 = plot_2d_2(
        time,
        time,
        z1,
        z2,
        title1=title1,
        title2=title2,
        xlabel2=xlabel,
        ylabel1=ylabel,
        ylabel2=ylabel,
        axis_x=axis_x,
        axis_y=axis_y,
    )
if plot_time and not plot_freq: #HERE
    plt.show()
    sys.exit()
###================================================================
### Plot functions in freq-domain
###================================================================

xlabel = "w1"
ylabel = "w2"
axis_x = [-4.0, 4.0]
axis_y = [-4.0, 4.0]
if plot_freq:

 if False:  # Q1 and Q2 factors

    title1 = "Q1"
    title2 = "Q2"

    fig4 = plot_2d_2(
        freq,
        freq,
        Q1,
        Q2,
        title1=title1,
        title2=title2,
        xlabel2=xlabel,
        ylabel1=ylabel,
        ylabel2=ylabel,
    )

 if False:  # Q+ and Q- factors
    title1 = "Q+"
    title2 = "Q-"
    legend = [title1, title2]

    fig5 = plot_2d_2(
        freq,
        freq,
        Qp,
        Qm,
        title1=title1,
        title2=title2,
        xlabel2=xlabel,
        ylabel1=ylabel,
        ylabel2=ylabel,
    )


    if False:
     idx = int(ndim / 2)
     data = [Qp[:, idx], Qm[:, idx]]
     print(idx, freq[idx])
     title = "cut at w2={}".format(freq[idx])
     fig5a = plot_1d_list(freq, data, title=title, xlabel="w", legend=legend)

 if False:

    title1 = "DKT_symm_fft.real"
    title2 = "DKT_asym_fft.imag "
    legend = [title1, title2]
    z1 = DKT_symm_fft.real
    z2 = DKT_asym_fft.imag

    fig6 = plot_2d_2(
        freq,
        freq,
        z1,
        z2,
        title1=title1,
        title2=title2,
        xlabel2=xlabel,
        ylabel1=ylabel,
        ylabel2=ylabel,
        axis_x=axis_x,
        axis_y=axis_y,
    )
    if False:
       idx = int(ndim / 2)
       data = [Resp_kubo[idx, :], Resp_symm[idx, :]]
       title = "cut at w1={}".format(freq[idx])
       fig6a = plot_1d_list(freq, data, title=title, xlabel="w", legend=legend)

 if False:
    title1 = "DKT_sym_fft.real"
    title2 = "DKT_sym_fft.imag"
    legend = [title1, title2]
    z1 = DKT_symm_fft.real
    z2 = DKT_symm_fft.imag

    fig6 = plot_2d_2(
        freq,
        freq,
        z1,
        z2,
        title1=title1,
        title2=title2,
        xlabel2=xlabel,
        ylabel1=ylabel,
        ylabel2=ylabel,
        axis_x=axis_x,
        axis_y=axis_y,
    )

 if False:
    title1 = "DKT_asym_fft.real"
    title2 = "DKT_asym_fft.imag"
    legend = [title1, title2]
    z1 = DKT_asym_fft.real
    z2 = DKT_asym_fft.imag

    fig6 = plot_2d_2(
        freq,
        freq,
        z1,
        z2,
        title1=title1,
        title2=title2,
        xlabel2=xlabel,
        ylabel1=ylabel,
        ylabel2=ylabel,
        axis_x=axis_x,
        axis_y=axis_y,
    )

 if False:
    title1 = "KTPB_ABC_fft.real"
    title2 = "KTPB_ABC_fft.imag"
    legend = [title1, title2]
    z1 = KTPB_ABC_fft.real
    z2 = KTPB_ABC_fft.imag

    fig6 = plot_2d_2(
        freq,
        freq,
        z1,
        z2,
        title1=title1,
        title2=title2,
        xlabel2=xlabel,
        ylabel1=ylabel,
        ylabel2=ylabel,
        axis_x=axis_x,
        axis_y=axis_y,
    )
 if False:
    title1 = "DKT_fft.real"
    title2 = "DKTp1_fft.imag"
    legend = [title1, title2]
    z1 = DKT_fft.real
    z2 = DKTp1_fft.imag

    fig6 = plot_2d_2(
        freq,
        freq,
        z1,
        z2,
        title1=title1,
        title2=title2,
        xlabel2=xlabel,
        ylabel1=ylabel,
        ylabel2=ylabel,
        axis_x=axis_x,
        axis_y=axis_y,
    )

 if False:
    title1 = "DKT_fft.real"
    title2 = "DKTp2_fft.imag"
    legend = [title1, title2]
    z1 = DKT_fft.real
    z2 = DKTp2_fft.imag

    fig6 = plot_2d_2(
        freq,
        freq,
        z1,
        z2,
        title1=title1,
        title2=title2,
        xlabel2=xlabel,
        ylabel1=ylabel,
        ylabel2=ylabel,
        axis_x=axis_x,
        axis_y=axis_y,
    )

 if False:
    title1 = "DKT_fft.real"
    title2 = "DKTpp_fft.real"
    legend = [title1, title2]
    z1 = DKT_fft.real
    z2 = DKTpp_fft.real

    fig6 = plot_2d_2(
        freq,
        freq,
        z1,
        z2,
        title1=title1,
        title2=title2,
        xlabel2=xlabel,
        ylabel1=ylabel,
        ylabel2=ylabel,
        axis_x=axis_x,
        axis_y=axis_y,
    )
 if True:
    title1 = "-beta KTPBp_ABC_fft.real"
    title2 = "beta^2 DKTpp_fft.real"
    legend = [title1, title2]
    z1=- beta * KTPBp_ABC_fft.real
    z2=(beta**2)* DKTpp_fft.real 

    fig6 = plot_2d_2(
        freq,
        freq,
        z1,
        z2,
        title1=title1,
        title2=title2,
        xlabel2=xlabel,
        ylabel1=ylabel,
        ylabel2=ylabel,
        axis_x=axis_x,
        axis_y=axis_y,
    )

    if True:
     legend.append('Resp-std')
     legend.append('test')
     idx = int(ndim / 2)
     print(legend)
     print(z1.shape,Resp_std.shape)
     data = [z1[:, idx], z2[:, idx],Resp_std[:,idx],z1[:, idx]+z2[:, idx]]
     print(idx, freq[idx])
     title = "cut at w2={}".format(freq[idx])
     fig5a = plot_1d_list(freq, data, title=title, xlabel="w", legend=legend)

     idx = int(ndim / 2)
     data = [z1[ idx,:], z2[ idx,:],Resp_std[idx,:],z1[ idx,:]+ z2[ idx,:]]
     print(idx, freq[idx])
     title = "cut at w1={}".format(freq[idx])
     fig5a = plot_1d_list(freq, data, title=title, xlabel="w", legend=legend)

 if False:
    title1 = "beta* DKTp1_fft.imag"
    title2 = "beta* DKTp2_fft.imag"
    legend = [title1, title2]
    z1=beta*DKTp1_fft.imag
    z2=beta*DKTp2_fft.imag 

    fig6 = plot_2d_2(
        freq,
        freq,
        z1,
        z2,
        title1=title1,
        title2=title2,
        xlabel2=xlabel,
        ylabel1=ylabel,
        ylabel2=ylabel,
        axis_x=axis_x,
        axis_y=axis_y,
    )


 if True:### Response functions `a la Kubo'

    title1 = "Resp_sym"
    title2 = "Resp_asym"
    legend = [title1, title2]

    fig6 = plot_2d_2(
    freq,
    freq,
    Resp_symm,
    Resp_asym,
    title1=title1,
    title2=title2,
    xlabel2=xlabel,
    ylabel1=ylabel,
    ylabel2=ylabel,
    axis_x=axis_x,
    axis_y=axis_y,
)

 if True: ### Resp_std vs Resp_new
    title1 = "Resp (eq.12)"
    title2 = "Resp_std"
    legend = [title1, title2]
    fig7 = plot_2d_2(
    freq,
    freq,
    Resp_new,
    Resp_std,
    title1=title1,
    title2=title2,
    xlabel2=xlabel,
    ylabel1=ylabel,
    ylabel2=ylabel,
    axis_x=axis_x,
    axis_y=axis_y,
)
 if True: ### Resp_std vs Resp_new
    title1 = "Resp (eq.11)"
    title2 = "Resp_std"
    legend = [title1, title2]

    fig7 = plot_2d_2(
    freq,
    freq,
    Resp_new2,
    Resp_std,
    title1=title1,
    title2=title2,
    xlabel2=xlabel,
    ylabel1=ylabel,
    ylabel2=ylabel,
    axis_x=axis_x,
    axis_y=axis_y,
)
 if True:### Response functions `a la Kubo'

    title1 = "Resp_kubo"
    title2 = "Resp_std"
    legend = [title1, title2]

    fig6 = plot_2d_2(
    freq,
    freq,
    Resp_kubo,
    Resp_std,
    title1=title1,
    title2=title2,
    xlabel2=xlabel,
    ylabel1=ylabel,
    ylabel2=ylabel,
    axis_x=axis_x,
    axis_y=axis_y,
)


 if False: #1D cuts
  idx = int(ndim / 2)
  data = [Resp_kubo[idx, :], Resp_symm[idx, :]]
  title = "cut at w1={}".format(freq[idx])
  fig6a = plot_1d_list(freq, data, title=title, xlabel="w", legend=legend)

  idx = np.argmin(np.abs(freq - f_target))
  print("freq target {} ,freq {}".format(f_target, freq[idx]))
  data = [Resp_kubo[idx, :], Resp_symm[idx, :]]
  title = "cut at w1={}".format(freq[idx])
  fig6b = plot_1d_list(freq, data, title=title, xlabel="w", legend=legend)

  idx = int(ndim / 2)
  data = [Resp_kubo[:, idx], Resp_symm[:, idx]]
  title = "cut at w2={}".format(freq[idx])
  fig6c = plot_1d_list(freq, data, title=title, xlabel="w", legend=legend)

  idx = np.argmin(np.abs(freq - f_target))
  # idx = np.argmin(np.abs(freq - 1.))
  data = [Resp_kubo[:, idx], Resp_symm[:, idx]]
  title = "cut at w2={}".format(freq[idx])
  fig6d = plot_1d_list(freq, data, title=title, xlabel="w", legend=legend)

  if False: ### Resp_kubo vs Resp_std
    title1 = "Resp_kubo"
    title2 = "Resp_std"
    legend = [title1, title2]

    fig7 = plot_2d_2(
    freq,
    freq,
    Resp_kubo,
    Resp_std,
    title1=title1,
    title2=title2,
    xlabel2=xlabel,
    ylabel1=ylabel,
    ylabel2=ylabel,
    axis_x=axis_x,
    axis_y=axis_y,
)
    if False:
     idx = int(ndim / 2)
     data = [Resp_kubo[idx, :], Resp_std[idx, :]]
     title = "cut at w1={}".format(freq[idx])
     fig6a = plot_1d_list(freq, data, title=title, xlabel="w", legend=legend)

     idx = np.argmin(np.abs(freq - f_target))
     # idx = np.argmin(np.abs(freq - 1.))
     data = [Resp_kubo[idx, :], Resp_std[idx, :]]
     title = "cut at w1={}".format(freq[idx])
     fig6b = plot_1d_list(freq, data, title=title, xlabel="w", legend=legend)

     idx = int(ndim / 2)
     data = [Resp_kubo[:, idx], Resp_std[:, idx]]
     title = "cut at w2={}".format(freq[idx])
     fig6c = plot_1d_list(freq, data, title=title, xlabel="w", legend=legend)

     idx = np.argmin(np.abs(freq - f_target))
     data = [Resp_kubo[:, idx], Resp_std[:, idx]]
     title = "cut at w2={}".format(freq[idx])
     fig6d = plot_1d_list(freq, data, title=title, xlabel="w", legend=legend)

###================================================================

###================================================================
### show plots

plt.show()

###================================================================

###================================================================
### save data to file
if False:
	### Resp_kubo: order is [freq,freq,Resp_kubo]
	output = open("Resp_kubo.dat", "w")
	for i in range(ndim):
	    for j in range(ndim):
	        output.write("{} {} {} \n".format(freq[i], freq[j], Resp_kubo[i, j]))
	    output.write("\n")
	output.close()

	### Resp_symm: order is [freq,freq,Resp_symm]
	output = open("Resp_symm.dat", "w")
	for i in range(ndim):
	    for j in range(ndim):
	        output.write("{} {} {} \n".format(freq[i], freq[j], Resp_symm[i, j]))
	    output.write("\n")
	output.close()

###================================================================
### End program
print("DONE!!")
