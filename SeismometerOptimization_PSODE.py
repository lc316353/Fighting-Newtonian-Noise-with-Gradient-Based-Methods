# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:57:14 2024

@author: schillings
"""

from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.single.general_optimizer import GeneralOptimizerPSO
from pyswarms.backend.topology import Pyramid
from pyswarms.backend.topology import Random
from pyswarms.backend.topology import VonNeumann
from pyswarms.backend.topology import Star
from pyswarms.backend.topology import Ring

from pyswarms.utils.plotters.formatters import Mesher
from pyswarms.utils.plotters import plot_contour
from pyswarms.utils.plotters.formatters import Designer

from scipy.optimize import differential_evolution

import numpy as np
from scipy import special as sp
from scipy import linalg
import time
import matplotlib.pyplot as plt

import json
from sys import argv

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

'''

Limit for cavity negligible respect to wavelength of the seismic field (ka->0). 
Seismometers with 3 measuring channels: x,y,z.
Test mass moving along e_TestMass

--Change geometrical setup, optimization method and loss right below here in the SETUP section
--Change the problem parameters N, f, p, SNR and some other things at the bottom of this program in the main()-function.
--Change hyperparameters of the optimization algorithms in the corresponding section

To be run like in a bash script:
	
    source [virtualEnvironmentPath]
	python3 [programName] [argument1] [argument2] ...
	
NOTE1: what the arguments are can be determined by setting parameters as type(argv[argumentNumber]) (see comments behind parameters in the main part)
NOTE2: you can also decomment in the main :
         #pool = Pool(processes=6)
        #pool.map(foo, range(30))
and 
        #lock = Lock() 
inside the foo () function

and also comment:
        foo(int(argv[2]))
to run this in parallel on the local pc.

'''
# ************************** SETUP **************************

# The geometry of ET: Directions of seismometers channel of measurment and mass test channel 
#e_TestMass1 = np.array([1,0,0])
#e_TestMass2 = np.array([0.5,np.sqrt(3)/2,0])
e_TestMass1 = np.array([np.sqrt(3)/2,-0.5,0])
e_TestMass2 = np.array([np.sqrt(3)/2,0.5,0])
d_TM_inx = 64.12 # m
d_TM_iny = 64.12 # m
d_TM_endx = 536.35 # m
d_TM_endy = 536.35 # m 


#cavern properties (only relevant for other coordinate systems)
cavern_length = 5000 #m
cavern_radius = 5 #m
reverse_cavern_length = 500 #m
cavern_radius_min = 5 #m
cavern_radius_max = 20 #m



#optimization settings
#mode="volume" ist the only one relevant for the paper, but you can do a lot more
mode="volume"                           #"volume", "sphere", "cylinder", "2cylinder", "2cylindervolume", add "forcesym" to force symmetry, add "multipleX" to adopt X seismometers per borehole (only for volume)
#switch between the metaheuristics
optimizationMethod = "particleSwarm"    #"particleSwarm", "differentialEvolution"
#single mirror optimizations or mean and max of them all
loss="all" #"end1", "end2", "in", "all"="max", "mean", 

#where to save your stuff
directory="mean"+"/"

Nmult=1
if "multiple" in mode.lower():
    try:
        Nmult=int(mode.split("multiple")[-1])
    except:
        print("Could not set Nmult!")

if "volume" in mode.lower():
    dim = 3+Nmult-1
else:
    dim = 2

if dim==2 and "sphere" in mode.lower():
    lower_bound = [0,0]
    upper_bound = [380,200]
elif dim==2 and "cylinder" in mode.lower() and not "2" in mode:
    lower_bound = [0,-reverse_cavern_length]
    upper_bound = [380,cavern_length]
elif dim==2 and "cylinder" in mode.lower() and "2" in mode:
    lower_bound = [-380,-reverse_cavern_length]
    upper_bound = [380,cavern_length]
elif dim==3 and "cylinder" in mode.lower() and "2" in mode:
    lower_bound = [-380,-reverse_cavern_length,cavern_radius_min]
    upper_bound = [380,cavern_length,cavern_radius_max]
elif dim==3:
    #choose bnd=1000 for most frequencies but bnd=5000 for very low frequencies close to 1 Hz
    bnd=1000.
    lower_bound = [-bnd,-bnd,-300.]
    upper_bound = [bnd,bnd,300.]
elif dim>3:
    bnd=5000
    lower_bound=[-bnd,-bnd]+[-300]*Nmult
    upper_bound=[bnd,bnd]+[300]*Nmult    
    
# ************************** SPECTRAL DENSITIES **************************

def CSS_3ch (kp, ks, N, x,y,z, SNR, p):
        # matrices with distances of each seismometer from the others for each coordinate
        if True:
            # optimizing, set zeros, only one loop, do not calculate diagonal elem but leave zero, antisymmetric part of the matrix
            mx = np.zeros([N,N])            
            my = np.zeros([N,N])
            mz = np.zeros([N,N])

            # triangular matrix without diag(zero!)
            for i in range(0,N):
                for j in range(i+1,N):
                    mx[i,j] = x[i]-x[j]
                    my[i,j] = y[i]-y[j]
                    mz[i,j] = z[i]-z[j]
                
            # remaining part
            mx=-mx.T+mx
            my=-my.T+my
            mz=-mz.T+mz
            #tmp = np.concatenate((mx.reshape(N*N,1),my.reshape(N*N,1), mz.reshape(N*N,1)),axis=1)
            #dist = squareform(pdist(tmp, metric='euclidean'))
            dist = np.sqrt(mx**2 + my**2 + mz**2)
        else:
            mx = pdist(x.reshape(N,1), lambda u, v: (u-v)) 
            my = pdist(y.reshape(N,1), lambda u, v: (u-v)) 
            mz = pdist(z.reshape(N,1), lambda u, v: (u-v))
            dist = squareform(np.sqrt(mx**2 + my**2 + mz**2))
            mx=squareform(mx)
            my=squareform(my)
            mz=squareform(mz)
        
        mask = dist == 0
        dist2 = dist.copy()
        dist2[mask] = 1 #zeros give problems in the division
        
        mx = mx/dist2
        my = my/dist2
        mz = mz/dist2
        
        #matrices for vector's dot operations:
        #3N x 3N matrices
        #(e1.e2) scalar product
        zo = np.zeros((N,N))
        o = np.ones((N,N))
        e1DoTe2 = np.concatenate((np.concatenate((o,zo,zo),1),np.concatenate((zo,o,zo),1),np.concatenate((zo,zo,o),1)),0)
        
        #(e1.e12) & (e2.e12) scalar products
        #3N x 3N matrices
        
        e1DoTe12 = np.concatenate((np.concatenate((mx,mx,mx),1),np.concatenate((my,my,my),1),np.concatenate((mz,mz,mz),1)),0)
        e2DoTe12 = np.concatenate((np.concatenate((mx,mx,mx),0),np.concatenate((my,my,my),0),np.concatenate((mz,mz,mz),0)),1)
        
        tmp = np.concatenate((dist,dist,dist),1)
        dist_m = np.concatenate((tmp,tmp,tmp),0)
        
        #tmp = np.concatenate((mask,mask,mask),1)
        #mask = np.concatenate((tmp,tmp,tmp),0)
        #del tmp
        
        fp = (sp.spherical_jn(0,dist_m*kp) + sp.spherical_jn(2,dist_m*kp))*e1DoTe2 - 3.*sp.spherical_jn(2,dist_m*kp)*e1DoTe12*e2DoTe12        
        # fp[mask] = 0        #Diagonal elements are 0 if the channel are perp. : see notes at p .103                                                                                                   
        dd = np.diag(np.ones(fp.shape[1],dtype=bool))
        fp[dd] = 1 + 1/(SNR)**2 #Diagonal elements are 1 : see notes at p .103
        
        
        fs = (sp.spherical_jn(0,dist_m*ks) - 0.5*sp.spherical_jn(2,dist_m*ks))*e1DoTe2 + (3./2)*sp.spherical_jn(2,dist_m*ks)*e1DoTe12*e2DoTe12                                                                                                                                                                                                                                           
        # fs[mask] = 0        #Diagonal elements are 0 if the channel are perp. : see notes at p .103 
        dd = np.diag(np.ones(fs.shape[1],dtype=bool))
        fs[dd] = 1 + 1/(SNR)**2 #Diagonal elements are 1 : see notes at p .103
        #SS:
        Css = p*fp + (1 - p)*fs
        #this correlates different channel noise of each seismometer
        #diagN = np.diag(N*[1])
        #snr = 0.1/(SNR**2)*np.concatenate((np.concatenate((zo, diagN, diagN), 1),np.concatenate((diagN, zo, diagN), 1), np.concatenate((diagN, diagN, zo), 1)),0 )
        Css = Css #+ snr
        return Css

    


def CSN_corr_funct (kp, ks, N, x,y,z, SNR, p, e_TestMass, d_TM1):

        # It is like having 3N sensors, 3 at each point, each one measuring along x,y,z
        xx  = np.concatenate((x,x,x), axis = 0)
        yy  = np.concatenate((y,y,y), axis = 0)
        zz  = np.concatenate((z,z,z), axis = 0)

        #es matrix
        mes = np.zeros((3,3*N))
        mes[0,0:N] = np.ones(N)
        mes[1,N:2*N] = np.ones(N)
        mes[2,2*N:3*N] = np.ones(N)


        e1 = np.array(e_TestMass)
        #es1 vector 

        es1 = np.ones((3,3*N))
        es1[0,:] = xx - d_TM1*e1[0]
        es1[1,:] = yy - d_TM1*e1[1]
        es1[2,:] = zz - d_TM1*e1[2]

        dist1 = np.sqrt(np.sum(es1**2, 0))
        es1 = es1/dist1 #normalization


        #e1 matrix
        me1 = np.ones((3,3*N))
        me1[0,:] = me1[0,:]*e1[0]
        me1[1,:] = me1[1,:]*e1[1]
        me1[2,:] = me1[2,:]*e1[2]

        #(es.es1) & (e1.es1) scalar products
        #e1 := test mass oscillation direction: e_TestMass

        esDoTes1 = np.sum(mes*es1,0)
        e1DoTes1 = np.sum(me1*es1,0)

        esDoTe1 = np.zeros(3)
        esDoTe1[0] = e1[0] # scalar product with x component of the sensor
        esDoTe1[1] = e1[1] # scalar product with y component of the sensor
        esDoTe1[2] = e1[2] # scalar product with z component of the sensor

        fp_s1 = np.zeros(3*N)
        fs_s1 = np.zeros(3*N)

        fp_s1[0:N] = (sp.spherical_jn(0,dist1[0:N]*kp) + sp.spherical_jn(2,dist1[0:N]*kp))*esDoTe1[0] - 3.*sp.spherical_jn(2,dist1[0:N]*kp)*esDoTes1[0:N]*e1DoTes1[0:N]	
        fs_s1[0:N] = (sp.spherical_jn(0,dist1[0:N]*ks) - 0.5*sp.spherical_jn(2,dist1[0:N]*ks))*esDoTe1[0] + (3./2)*sp.spherical_jn(2,dist1[0:N]*ks)*esDoTes1[0:N]*e1DoTes1[0:N]												          																   

        fp_s1[N:2*N] = (sp.spherical_jn(0,dist1[N:2*N]*kp) + sp.spherical_jn(2,dist1[N:2*N]*kp))*esDoTe1[1] - 3.*sp.spherical_jn(2,dist1[N:2*N]*kp)*esDoTes1[N:2*N]*e1DoTes1[N:2*N]	
        fs_s1[N:2*N] = (sp.spherical_jn(0,dist1[N:2*N]*ks) - 0.5*sp.spherical_jn(2,dist1[N:2*N]*ks))*esDoTe1[1] + (3./2)*sp.spherical_jn(2,dist1[N:2*N]*ks)*esDoTes1[N:2*N]*e1DoTes1[N:2*N]												          																   

        fp_s1[2*N:3*N] = (sp.spherical_jn(0,dist1[2*N:3*N]*kp) + sp.spherical_jn(2,dist1[2*N:3*N]*kp))*esDoTe1[2] - 3.*sp.spherical_jn(2,dist1[2*N:3*N]*kp)*esDoTes1[2*N:3*N]*e1DoTes1[2*N:3*N]	
        fs_s1[2*N:3*N] = (sp.spherical_jn(0,dist1[2*N:3*N]*ks) - 0.5*sp.spherical_jn(2,dist1[2*N:3*N]*ks))*esDoTe1[2] + (3./2)*sp.spherical_jn(2,dist1[2*N:3*N]*ks)*esDoTes1[2*N:3*N]*e1DoTes1[2*N:3*N]												          																   

        return fp_s1, fs_s1

def CSN_END_ch3 (kp, ks, N, x,y,z, SNR, p, e_TestMass, d_TM):

        fp_s1, fs_s1 =  CSN_corr_funct (kp, ks, N, x,y,z, SNR, p, e_TestMass, d_TM)

        #SN:
        Csn = 1/3*(2*p*fp_s1 - (1-p)*fs_s1) #multiplied by (4 pi rho_0 G) S_tot which simplifies with Css and Cnn

        return Csn

def CSN_2IN_ch3 (kp, ks, N, x,y,z, SNR, p, e_TestMass1, e_TestMass2, d_TM1, d_TM2):

        fp_s1, fs_s1 =  CSN_corr_funct (kp, ks, N, x,y,z, SNR, p, e_TestMass1, d_TM1)
        fp_s2, fs_s2 =  CSN_corr_funct (kp, ks, N, x,y,z, SNR, p, e_TestMass2, d_TM2)

        #SN:
        Csn = 1/3*(2*p*(fp_s2 - fp_s1) - (1-p)*(fs_s2 - fs_s1)) #multiplied by (4 pi rho_0 G) S_tot which simplifies with Css and Cnn
        return Csn

def CNN_3ch (p):
        #NN:
        Cnn = 1/9*(3*p + 1) #multiplied by (4 pi rho_0 G)^2 S_tot which simplifies with Csn and Css
        return Cnn

def CNN_2IN_3ch (p, kp, ks, e_TestMass1, e_TestMass2, d_TM1, d_TM2):
        #NN:
        e2DoTe1 = np.dot(e_TestMass1,e_TestMass2)
        e21 = d_TM1*e_TestMass1-d_TM2*e_TestMass2
        dist1 = np.linalg.norm((e21))
        e21 = e21/dist1
        e2DoTe21 = np.dot(e_TestMass2,e21)
        e1DoTe21 = np.dot(e_TestMass1,e21)
        
        fp = (sp.spherical_jn(0,dist1*kp) + sp.spherical_jn(2,dist1*kp))*e2DoTe1 - 3.*sp.spherical_jn(2,dist1*kp)*e2DoTe21*e1DoTe21
        fs = (sp.spherical_jn(0,dist1*ks) - 0.5*sp.spherical_jn(2,dist1*ks))*e2DoTe1 + (3./2)*sp.spherical_jn(2,dist1*ks)*e2DoTe21*e1DoTe21
        Cnn = 1/9*(2*(3*p + 1) - 2*(4*p*fp + (1-p)*fs))  #multiplied by (4 pi rho_0 G)^2 S_tot which simplifies with Csn and Css
        return Cnn


def CSS_svd_3ch(Css):

    Diag = np.diag(Css)
    Nfact = np.sqrt(np.tensordot(Diag,Diag, axes = 0))
    """
    Nfact is a matrix where each elements is Nfact_ij = ASD_i ASD_j, with ASD the amplitude spectral density of the sensor i or j
    In the end Css become a coherence matrix 
    """
    Css = Css/Nfact
    
    """
    why the svd: 
    with the singular value decomposition, you can write any matrix M as M = UEV and since U and V are 
    unitary, the inverse of M is easy to be calculated, iM is: iM = hU iE hV, where hU and hV are the 
    transposed conjugate of U and V, respectively. iE is easy to calculate since E is diagonal.
    If for some numerical problems you cannot calculate the inverse, this means that E must have some 
    diagonal value close to zero. You can remove these values from E and cut U and V accordingly to 
    reconstruct the pseudo-inverse of M. So, it is a trick to avoid problems in the inversion. 
    That function returns already the inverse of M.
    """
    
    [U,diagS,V] = linalg.svd(Css)
    #		S = np.diag(diagS)
    thresh = 0.01         
    kVal = np.sum(diagS > thresh)
    #		Css_svd = (U[:, 0: kVal]@np.diag(diagS[0: kVal])@V[0: kVal, :])#inverse of the reconstructed Css
    iU = (U.conjugate()).transpose()
    iV = (V.conjugate()).transpose()
    Css_svd = (iV[:,0: kVal])@np.diag(1/diagS[0: kVal])@(iU[0: kVal,:])#inverse of the reconstructed Css
    Css_svd = Css_svd/Nfact
    
    return Css_svd

''' ********************************************************************************************************************************** '''


# ************************** RESIDUAL FUNCTION **************************

def Residual (state, N, freq, SNR, p, mirror="all"):
        
        state = np.array(state)
    
        kp = 2*np.pi*freq/6000 #velocity for p-waves 6000 m/s
        ks = 2*np.pi*freq/4000 #velocity for s-waves 4000 m/s
        
        #coordinate of each seismometer and create matrix of distances between each seismometers
        if "differentialevolution" in optimizationMethod.lower():
            state=np.array([state])
        n_part = state.shape[0]
        Res_Vec = np.zeros(n_part)
        
        Nloc=N
        for tt in range(0,n_part):
            #recalculate different coordinates to cartesian
            s = state[tt,:].reshape(N,dim)
            if dim==2 and "sphere" in mode.lower():
                phi = s[:,0]*np.pi/180
                theta = s[:,1]*np.pi/180
                x = cavern_radius*np.sin(theta)*np.cos(phi)
                y = cavern_radius*np.sin(theta)*np.sin(phi)
                z = cavern_radius*np.cos(theta)
            elif dim==2 and "cylinder" in mode.lower() and not "2" in mode:
                phi = s[:,0]*np.pi/180
                x = s[:,1]
                y = cavern_radius*np.cos(phi)
                z = cavern_radius*np.sin(phi)
            elif dim==2 and "cylinder" in mode.lower() and "2" in mode:
                e1 = e_TestMass1[0]*(s[:,0]>0)+e_TestMass2[0]*(s[:,0]<=0)
                e2 = e_TestMass1[1]*(s[:,0]>0)+e_TestMass2[1]*(s[:,0]<=0)
                e_perp1 = np.cross(e_TestMass1,np.array([0,0,1]))[0]*(s[:,0]>0) - np.cross(e_TestMass2,np.array([0,0,1]))[0]*(s[:,0]<=0)
                e_perp2 = np.cross(e_TestMass1,np.array([0,0,1]))[1]*(s[:,0]>0) - np.cross(e_TestMass2,np.array([0,0,1]))[1]*(s[:,0]<=0)
                phi = s[:,0]*np.pi/180*(s[:,0]>0) - s[:,0]*np.pi/180*(s[:,0]<=0)
                x = s[:,1]*e1 + cavern_radius*np.cos(phi)*e_perp1
                y= s[:,1]*e2 + cavern_radius*np.cos(phi)*e_perp2
                z = cavern_radius*np.sin(phi)
            elif dim==3 and "cylinder" in mode.lower() and "2" in mode:
                e1 = e_TestMass1[0]*(s[:,0]>0)+e_TestMass2[0]*(s[:,0]<=0)
                e2 = e_TestMass1[1]*(s[:,0]>0)+e_TestMass2[1]*(s[:,0]<=0)
                e_perp1 = np.cross(e_TestMass1,np.array([0,0,1]))[0]*(s[:,0]>0) - np.cross(e_TestMass2,np.array([0,0,1]))[0]*(s[:,0]<=0)
                e_perp2 = np.cross(e_TestMass1,np.array([0,0,1]))[1]*(s[:,0]>0) - np.cross(e_TestMass2,np.array([0,0,1]))[1]*(s[:,0]<=0)
                phi = s[:,0]*np.pi/180*(s[:,0]>0) - s[:,0]*np.pi/180*(s[:,0]<=0)
                x = s[:,1]*e1 + s[:,2]*np.cos(phi)*e_perp1
                y= s[:,1]*e2 + s[:,2]*np.cos(phi)*e_perp2
                z = s[:,2]*np.sin(phi)
            elif dim==3:
                x = s[:,0]
                y = s[:,1]
                z = s[:,2]    
            elif dim>3:
                x = np.array(list(s[:,0])*Nmult)
                y = np.array(list(s[:,1])*Nmult)
                z = np.concatenate(s[:,2:].T)
                Nloc=N*Nmult
                
            if "forcesym" in mode.lower():
                Nloc=2*N
                x=np.concatenate((x, x))
                y=np.concatenate((y, -y))
                z=np.concatenate((z, z))
                
            #CPSDs
            """************************* correlation between seismometers calculation: ******************""" 
            Css = CSS_3ch(kp, ks, Nloc, x,y,z, SNR, p)
            """****************** correlation between seismometers and test mass calculation: **************"""
            Csn_in = CSN_2IN_ch3(kp, ks, Nloc, x,y,z, SNR, p, e_TestMass1, e_TestMass2, d_TM_inx, d_TM_iny)
            Csn_end1 = CSN_END_ch3(kp, ks, Nloc, x,y,z, SNR, p, e_TestMass1, d_TM_endx)
            Csn_end2 = CSN_END_ch3(kp, ks, Nloc, x,y,z, SNR, p, e_TestMass2, d_TM_endy)
            """****************** correlation of the test mass: **************"""
            Cnn_END = CNN_3ch(p)
            Cnn_2IN = CNN_2IN_3ch (p, kp, ks, e_TestMass1, e_TestMass2, d_TM_inx, d_TM_iny)
    
    
            """ ************* RESIDUAL CALCULATION ***********************"""
            Csn = [Csn_in, Csn_end1, Csn_end2]
            Cnn = [Cnn_2IN, Cnn_END, Cnn_END]
            nn = len(Csn)
            Res_v = np.zeros(nn)
            
            for rr in range(0,nn):
                X = linalg.solve(Css, Csn[rr], assume_a="gen", overwrite_a=False)
                resid = 1-np.dot(Csn[rr],X)/Cnn[rr]
                if (resid < 0):
                    print("NEGATIVE RESIDUAL", resid, "-- rr=", rr)
                    Css_svd = CSS_svd_3ch(Css)
                    resid = 1 - np.dot(Csn[rr].conjugate(),np.dot(Css_svd,Csn[rr]))/Cnn[rr]
                    # print('residual ', resid)
                Res_v[rr] = resid
            
            #residual summary
            if mirror=="in":
                Res_Vec[tt] = Res_v[0]
            elif mirror=="end1":
                Res_Vec[tt] = Res_v[1]
            elif mirror=="end2":
                Res_Vec[tt] = Res_v[2]
            elif mirror=="mean" or mirror=="all":
                Res_Vec[tt] = np.mean(Res_v)
            elif mirror=="max":
                Res_Vec[tt] = np.max(Res_v)
            else:
                Res_Vec[tt] = np.max(Res_v)
        if "particleswarm" in optimizationMethod.lower():
            return Res_Vec
        else:
            return Res_Vec[0]



"""*********************************************************   MAIN   ******************************************************************"""



def foo(N=10, freq=1, SNR=15, p=0.2, mirror="all", ID=0, savename="Results", worker=1, animate=False):
                
        print('starting proc ...')
        starttime=time.time()
        
        

        # ************************** PARTICLE SWARM ALGORITHM **************************

       	# initiate the optimizer
        if "particleswarm" in optimizationMethod.lower():
            niter = 1000
            ftol = 1e-3 #precision of early stopping
            ftol_iter = 20 #patience of early stopping
            
            x_min=np.tile(lower_bound, N)
            x_max=np.tile(upper_bound, N)
            bounds = (x_min, x_max)
            
            swarm_size = 800 #For 4000 choose 4000 MB memory
            options = {'c1': 1.5, 'c2': 2, 'w': 0.1, 'k': swarm_size/10, 'p': 2}
            optimizer = GeneralOptimizerPSO(n_particles=swarm_size, dimensions=dim*N, options=options, bounds=bounds, ftol = ftol,ftol_iter = ftol_iter, topology=Ring())
                    
            
            Final_State = optimizer.optimize(Residual, niter, n_processes=worker, N=N, freq=freq, SNR=SNR, p=p, mirror=mirror)
            best_R = Final_State[0]
            best_x = Final_State[1]
            
            
        # **************************** DIFFERENTIAL EVOLUTION ALGORITHM **************************                        
        elif "differentialevolution" in optimizationMethod.lower():

            #It serves to the DE algorithm to search the minimum inside the boundaries (each dimension have to have the boundary)
            bound = np.array([lower_bound,upper_bound]).T #[(-100., 700.), (-300., 700.), (-300., 300.)]
            x_bound = list(bound)*N
            
            
            #Parameters to be passed to the Residual function
            residualParameter = (N, freq, SNR, p, mirror)
            
            niter = 4500
            ftol = 1e-3
            
            popsize = 65
            recombination = 0.75
            mutation = (0, 1.5)

            Final_State = differential_evolution(Residual, x_bound, residualParameter, disp=True, maxiter=niter, popsize=popsize, init='random', workers=worker, recombination=recombination, mutation=mutation, strategy='best1bin', tol=ftol, updating='deferred')
            optimizer = Final_State
            best_x = Final_State.x
            best_R = Residual(Final_State.x,*residualParameter)
        
        
        best_x = best_x.reshape(N,dim)
        print('\n',best_R,'\n',best_x,'\n')
        
        # ************************** VISUALIZATION **************************
        
        if animate:
            plt.close("all")
            
            #optimization visualization
            if "particleswarm" in optimizationMethod.lower():
                pos_hist=np.array(optimizer.pos_history)
                def Residualgiven(state,N=N,freq=freq,SNR=SNR,p=p, mirror=mirror):
                    return Residual (state, N, freq, SNR, p, mirror)
                fig,ax=plt.subplots()
                #ax.set_xlim(x_min[0],x_max[0])
                #ax.set_ylim(x_min[1],x_max[1])
                if dim==3:
                    #d = Designer(limits=[(x_min[0],x_max[0]),(x_min[1],x_max[1]),(x_min[2],x_max[2])], label=['no. 0, x', 'no. 0, y', 'no. 0, z'])
                    d = Designer(limits=[(x_min[1],x_max[1]),(x_min[2],x_max[2])], label=['no. 0, y', 'no. 0, z'])
                    animation = plot_contour(pos_history=pos_hist[:,:,1:3],designer=d,canvas=(fig,ax), mark=(0,0))
                else:
                    d = Designer(limits=[(x_min[0],x_max[0]),(x_min[1],x_max[1])], label=['no. 0, coord 1', 'no. 0, coord 2'])
                    animation = plot_contour(pos_history=pos_hist[:,:,:2],designer=d,canvas=(fig,ax), mark=(0,0))
                animation.save('animateSwarm'+str(ID)+'.gif', fps=10)
            
                plt.figure()
                plt.title("some swarm particle seismometer no. 0 x positions")
                plt.xlabel("iteration")
                plt.ylabel("x")
                for i in [2,10,14,18,33]: 
                    plt.plot(pos_hist[:,i,0])

                plt.figure()
                plt.title("The seismometer positions during optimization")
                plt.xlabel("iteration")
                plt.ylabel("x,y,z")
                for j in range(dim*min(N,3)): 
                    plt.plot(pos_hist[:,0,j],label="no. {}, {}-pos".format(j//dim,(["x","y"]+["z"+str(j-2)]*Nmult)[j%dim]))
                plt.legend()
            
            #3d position plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            if dim==2 and "sphere" in mode.lower():
                phi = best_x[:,0]*np.pi/180
                theta = best_x[:,1]*np.pi/180
                bx = cavern_radius*np.sin(theta)*np.cos(phi)
                by = cavern_radius*np.sin(theta)*np.sin(phi)
                bz = cavern_radius*np.cos(theta)
            elif dim==2 and "cylinder" in mode.lower() and not "2" in mode:
                phi = best_x[:,0]*np.pi/180
                bx = best_x[:,1]
                by = cavern_radius*np.cos(phi)
                bz = cavern_radius*np.sin(phi)
            elif dim==2 and "cylinder" in mode.lower() and "2" in mode:
                e1 = e_TestMass1[0]*(best_x[:,0]>0)+e_TestMass2[0]*(best_x[:,0]<=0)
                e2 = e_TestMass1[1]*(best_x[:,0]>0)+e_TestMass2[1]*(best_x[:,0]<=0)
                e_perp1 = np.cross(e_TestMass1,np.array([0,0,1]))[0]*(best_x[:,0]>0) - np.cross(e_TestMass2,np.array([0,0,1]))[0]*(best_x[:,0]<=0)
                e_perp2 = np.cross(e_TestMass1,np.array([0,0,1]))[1]*(best_x[:,0]>0) - np.cross(e_TestMass2,np.array([0,0,1]))[1]*(best_x[:,0]<=0)
                phi = best_x[:,0]*np.pi/180*(best_x[:,0]>0) - best_x[:,0]*np.pi/180*(best_x[:,0]<=0)
                bx = best_x[:,1]*e1 + cavern_radius*np.cos(phi)*e_perp1
                by= best_x[:,1]*e2 + cavern_radius*np.cos(phi)*e_perp2
                bz = cavern_radius*np.sin(phi)
            elif dim==3 and "cylinder" in mode.lower() and "2" in mode:
                e1 = e_TestMass1[0]*(best_x[:,0]>0)+e_TestMass2[0]*(best_x[:,0]<=0)
                e2 = e_TestMass1[1]*(best_x[:,0]>0)+e_TestMass2[1]*(best_x[:,0]<=0)
                e_perp1 = np.cross(e_TestMass1,np.array([0,0,1]))[0]*(best_x[:,0]>0) - np.cross(e_TestMass2,np.array([0,0,1]))[0]*(best_x[:,0]<=0)
                e_perp2 = np.cross(e_TestMass1,np.array([0,0,1]))[1]*(best_x[:,0]>0) - np.cross(e_TestMass2,np.array([0,0,1]))[1]*(best_x[:,0]<=0)
                phi = best_x[:,0]*np.pi/180*(best_x[:,0]>0) - best_x[:,0]*np.pi/180*(best_x[:,0]<=0)
                bx = best_x[:,1]*e1 + best_x[:,2]*np.cos(phi)*e_perp1
                by= best_x[:,1]*e2 + best_x[:,2]*np.cos(phi)*e_perp2
                bz = best_x[:,2]*np.sin(phi)
            elif dim==3:
                bx = best_x[:,0]
                by = best_x[:,1]
                bz = best_x[:,2]
            elif dim>3:
                bx = np.array(list(best_x[:,0])*Nmult)
                by = np.array(list(best_x[:,1])*Nmult)
                bz = np.concatenate(best_x[:,2:].T)
                
                
            if "forcesym" in mode.lower():
                bx = np.concatenate((bx, bx))
                by = np.concatenate((by, -by))
                bz = np.concatenate((bz, bz))
                
            ax.scatter(bx, by, bz, c='g', marker='o')
            
            ax.scatter(d_TM_inx*e_TestMass1[0],d_TM_inx*e_TestMass1[1],d_TM_inx*e_TestMass1[2],c='r', marker='o')
            ax.scatter(d_TM_endx*e_TestMass1[0],d_TM_endx*e_TestMass1[1],d_TM_endx*e_TestMass1[2],c='r', marker='o')
            ax.scatter(d_TM_iny*e_TestMass2[0],d_TM_iny*e_TestMass2[1],d_TM_iny*e_TestMass2[2],c='r', marker='o')
            ax.scatter(d_TM_endy*e_TestMass2[0],d_TM_endy*e_TestMass2[1],d_TM_endy*e_TestMass2[2],c='r', marker='o')
            ax.plot([-reverse_cavern_length*e_TestMass1[0],cavern_length*e_TestMass1[0]], [-reverse_cavern_length*e_TestMass1[1],cavern_length*e_TestMass1[1]], '--', c = 'k')
            ax.plot([-reverse_cavern_length*e_TestMass2[0],cavern_length*e_TestMass2[0]], [-reverse_cavern_length*e_TestMass2[1],cavern_length*e_TestMass2[1]], '--', c = 'k')
        
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_title(r"$\sqrt{R}$ = "+str(np.sqrt(best_R)))
            
            
        # ************************** WRITING FILE **************************
        
        filename = savename+str(ID)+'.txt'
        f = open(directory+filename,'a+') 
          
        #log all important information and results
        #In principle you can copy this into the I-python-console to see the positions but works only for mode=="volume"
        f.write('\n \n \n## *************'+optimizationMethod+'-'+mode+"-"+mirror+": "+str(ID) + '*************** ##\n \n \n' )
        
        f.write('import numpy as np\n')
        f.write('import matplotlib.pyplot as plt\n')
        f.write('fig = plt.figure()\n')
        f.write('ax = fig.add_subplot(111, projection=\'3d\')\n')

        f.write('p = ')
        json.dump(p, f)
        f.write('\n')

        f.write('SNR = ')
        json.dump(SNR, f)
        f.write('\n')

        f.write('N = ')
        json.dump(N,f)
        f.write('\n')

        f.write('f = ')
        json.dump(freq,f)
        f.write('\n')
        
        f.write('bounds = np.array(['+str(lower_bound)+','+str(upper_bound)+'])\n')
        if "particleswarm" in optimizationMethod.lower():
            f.write('optimizer_options = {"swarm_size": '+str(swarm_size)+', "c1": '+str(options['c1'])+', "c2": '+str(options['c2'])+', "w": '+str(options['w'])+', "k": '+str(options['k'])+', "p": '+str(options['p'])+'}\n')
        elif "differentialevolution" in optimizationMethod.lower():
            f.write('optimizer_options = {"NP": '+str(popsize)+', "CR": '+str(recombination)+', "F": '+str(mutation)+'}\n')
        else:
            f.write("#undefined optimizer\n")
            
        f.write('Energy = ') #I do not now about the meaning of 'Energy' for sqrt(R) but I got used to it
        # remember to take the root, skipped in the optimizing process
        json.dump(np.sqrt(best_R),f)
        
        f.write('\ne2 = ')
        json.dump(d_TM_iny,f)
        f.write('*np.array(['+str(e_TestMass2[0])+','+str(e_TestMass2[1])+','+str(e_TestMass2[2])+'])')
        f.write('\ne1 = ')
        json.dump(d_TM_inx, f)
        f.write('*np.array(['+str(e_TestMass1[0])+','+str(e_TestMass1[1])+','+str(e_TestMass1[2])+'])')

        f.write('\ne3 = ')
        json.dump(d_TM_endy,f)
        f.write('*np.array(['+str(e_TestMass2[0])+','+str(e_TestMass2[1])+','+str(e_TestMass2[2])+'])')
        f.write('\ne4 = ')
        json.dump(d_TM_endx, f)
        f.write('*np.array(['+str(e_TestMass1[0])+','+str(e_TestMass1[1])+','+str(e_TestMass1[2])+'])')
        f.write('\n')

        aa = 'FinalState'+str(ID)+' = np.array('
        f.write(aa)
        json.dump(best_x.tolist() , f) 
        f.write(')\n')
        aa = 'FinalState'+str(ID) + ' = FinalState'+str(ID)+'.reshape(N,3)\n'
        f.write(aa)
        aa = 'ax.scatter(FinalState'+str(ID)+'[:,0],FinalState'+str(ID)+'[:,1],FinalState'+str(ID)+'[:,2],c=\'g\', marker=\'o\')\n'
        f.write(aa)
        aa = 'ax.scatter(e1[0],e1[1],e1[2],c=\'r\', marker=\'o\')\n'
        f.write(aa)
        aa = 'ax.scatter(e2[0],e2[1],e2[2],c=\'r\', marker=\'o\')\n'
        f.write(aa)
        aa = 'ax.scatter(e3[0],e3[1],e3[2],c=\'r\', marker=\'o\')\n'
        f.write(aa)
        aa = 'ax.scatter(e4[0],e4[1],e4[2],c=\'r\', marker=\'o\')\n'
        f.write(aa)
        
        aa = 'plt.plot([0,e4[0]], [0,e4[1]], \'--\', c=\'k\')\nplt.plot([0, e3[0]], [0, e3[1]], \'--\', c = \'k\')\n'
        f.write(aa)
        f.write('plt.show()\n')
        aa = 'plt.xlabel(\'x\')\n'
        f.write(aa)
        aa = 'plt.ylabel(\'y\')\n'
        f.write(aa)
        aa = 'ax.set_zlabel(\'z\')\n'
        f.write(aa)
        aa = 'ax.set_title(\'Energie=\'+str(np.round(Energy,4)))\n'
        f.write(aa)
        
        f.write("#Finished in "+str(np.round((time.time()-starttime)/60,2))+" minutes")

        f.close()       
        
        print("Finished in",np.round((time.time()-starttime)/60,2),"minutes")
  
        
        return optimizer
    
    
    
    
    
    

        # ************************** MAIN **************************

if __name__ == '__main__':
        
        #number of seismometers (At the end it's like having N*3 seismometers since each seismometer is composed by three channels (x,y,z): like having 3 seismometers in N positions)
        N = int(argv[1])
        
        #wiener filter frequency
        freq = 10

        #signal to noise ratio
        SNR = 15

        #mixing ratio (p*100% of P and (1-p)*100% of S; p is in the interval [0,1])
        p = 0.2
        
        #mirror relevant for loss
        mirror = loss
        
        #job identifier (part of result file name)
        ID = N #int(argv[2])
    
        #name of numerated Savefiles
        savename = str(argv[2]) #str(argv[3])
        
        #do plots and vizualization (should be off on cluster)
        animate=False
        
        #optional parallelization
        ww = 1 #int(argv[4]) #workers
        #pool = Pool(processes=6)
        #pool.map(foo, range(30))
    
        optimizer=foo(N, freq, SNR, p, mirror, ID, savename, ww, animate)






