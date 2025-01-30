# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 11:49:27 2024

@author: schillings
"""

#import numpy as np
import jax_bessel
from jax.scipy import linalg
import time
import matplotlib.pyplot as plt

import json
from sys import argv
import ast

import numpy as onp
import jax.numpy as np
import jax.scipy.optimize as opt
import scipy.optimize as oopt
import optax
import jax
import jaxopt

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from functools import partial


'''

Limit for cavity negligible respect to wavelength of the seismic field (ka->0). 
Seismometers with 3 measuring channels: x,y,z.
Test mass moving along e_TestMass

--Change geometrical setupand problem parameters right below here in the SETUP section
--Change hyperparameters of the optimization algorithms in the corresponding section

To be run like in a bash script:
	
    source [virtualEnvironmentPath]
	python3 [programName] [argument1] [argument2] ...
	
NOTE1: what the arguments are can be determined by setting parameters as type(argv[argumentNumber]) (see comments behind parameters in the main part)

'''
# ************************** SETUP **************************

#The geometry of ET: Directions of seismometrs channel of measurment and mass test channel 
        
#e_TestMass1 = np.array([1,0,0])
#e_TestMass2 = np.array([0.5,np.sqrt(3)/2,0])
e_TestMass1 = np.array([np.sqrt(3)/2,-0.5,0])
e_TestMass2 = np.array([np.sqrt(3)/2,0.5,0])
d_TM_inx = 64.12 # m
d_TM_iny = 64.12 # m
d_TM_endx = 536.35 # m
d_TM_endy = 536.35 # m 

#for some different geometry
cavern_radius=5 #m
dim=3

    
#Number of seismometers
N = int(argv[1]) 

#Job identifier
ID = N #int(100*argv[1]) 

#Should additional plots be produced
animate=False

#Parameters definitions
freq = 10

#Signal to Noise Ratio
SNR = 15

#Mixing ratio (p*100% of S and (1-p)*100% of P; p is in the interval [0,1])
p = 0.2

#Mirror to optimize positions for
mirror="all" #savename.split("Results")[1]

#where to look for the pre-optimized file
folder="../FrancUnified/mean/" #"../Ndependence/sym/"

#what tag you have named the pre-optimized file with.
#Typically, we used "[N][tag]Resultsall[ID].txt" which will be read for the given tag
tag="PSOmeancorNoise" #"MC"


#How to save the new files
savename = "adam"+"Results" #str(argv[3]) 
savename="newtest/"+str(N)+tag+savename + mirror

#only "adam" works, really
whichOptimizer="adam" #"adam", "lbfgs", "scipy"

#if adding a barrier like np.exp(a*np.sum((z/600)**expon)) to the optimization. Else write a=0
a=int(argv[2])
expon=int(argv[3])

#this should be correctly read if files were generated with SeismometerOptimization_PSODE.py or this code
energyline=17
stateline=21
timeline=36
blocksize=35

# ************************** SPECTRAL DENSITIES **************************

@partial(jax.jit,static_argnums=(2,))
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
                    mx=mx.at[i,j].set(x[i]-x[j])
                    my=my.at[i,j].set(y[i]-y[j])
                    mz=mz.at[i,j].set(z[i]-z[j])
                
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
        
        #mask = dist == 0
        #dist2 = dist.copy()
        #dist2=dist2.at[mask].set(1) #zeros give problems in the division
        
        #mx = mx/dist2
        #my = my/dist2
        #mz = mz/dist2
        
        
        mx = mx/dist
        my = my/dist
        mz = mz/dist
        
        mx=np.nan_to_num(mx,nan=0.0)
        my=np.nan_to_num(my,nan=0.0)
        mz=np.nan_to_num(mz,nan=0.0)
        
        
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
        
        fp = (jax_bessel.spherical_jv(0,dist_m*kp) + jax_bessel.spherical_jv(2,dist_m*kp))*e1DoTe2 - 3.*jax_bessel.spherical_jv(2,dist_m*kp)*e1DoTe12*e2DoTe12        
        # fp[mask] = 0        #Diagonal elements are 0 if the channel are perp. : see notes at p .103                                                                                                   
        dd = np.diag(np.ones(fp.shape[1],dtype=bool))
        fp = np.where(dd,1 + 1/(SNR)**2,fp)     #fp.at[dd].set(1 + 1/(SNR)**2) #Diagonal elements are 1 : see notes at p .103
        
        fs = (jax_bessel.spherical_jv(0,dist_m*ks) - 0.5*jax_bessel.spherical_jv(2,dist_m*ks))*e1DoTe2 + (3./2)*jax_bessel.spherical_jv(2,dist_m*ks)*e1DoTe12*e2DoTe12                                                                                                                                                                                                                                           
        # fs[mask] = 0        #Diagonal elements are 0 if the channel are perp. : see notes at p .103 
        dd = np.diag(np.ones(fs.shape[1],dtype=bool))
        fs = np.where(dd,1 + 1/(SNR)**2,fs)     #fs.at[dd].set(1 + 1/(SNR)**2) #Diagonal elements are 1 : see notes at p .103
        #SS:
        Css = p*fp + (1 - p)*fs
        diagN = np.diag(np.array(N*[1]))
        snr = 0.1/((SNR)**2)*np.concatenate((np.concatenate((zo, diagN, diagN), 1),np.concatenate((diagN, zo, diagN), 1), np.concatenate((diagN, diagN, zo), 1)),0 )
        Css = Css + snr
        return Css

    

@partial(jax.jit,static_argnums=(2,))
def CSN_corr_funct (kp, ks, N, x,y,z, SNR, p, e_TestMass, d_TM1):

        # It is like having 3N sensors, 3 at each point, each one measuring along x,y,z
        xx  = np.concatenate((x,x,x), axis = 0)
        yy  = np.concatenate((y,y,y), axis = 0)
        zz  = np.concatenate((z,z,z), axis = 0)

        #es matrix
        mes = np.zeros((3,3*N))
        mes=mes.at[0,0:N].set(np.ones(N))
        mes=mes.at[1,N:2*N].set(np.ones(N))
        mes=mes.at[2,2*N:3*N].set(np.ones(N))


        e1 = np.array(e_TestMass)
        #es1 vector 

        es1 = np.ones((3,3*N))
        es1=es1.at[0,:].set(xx - d_TM1*e1[0])
        es1=es1.at[1,:].set(yy - d_TM1*e1[1])
        es1=es1.at[2,:].set(zz - d_TM1*e1[2])

        dist1 = np.sqrt(np.sum(es1**2, 0))
        es1 = es1/dist1 #normalization


        #e1 matrix
        me1 = np.ones((3,3*N))
        me1=me1.at[0,:].set(me1[0,:]*e1[0])
        me1=me1.at[1,:].set(me1[1,:]*e1[1])
        me1=me1.at[2,:].set(me1[2,:]*e1[2])

        #(es.es1) & (e1.es1) scalar products
        #e1 := test mass oscillation direction: e_TestMass

        esDoTes1 = np.sum(mes*es1,0)
        e1DoTes1 = np.sum(me1*es1,0)

        esDoTe1 = np.zeros(3)
        esDoTe1=esDoTe1.at[0].set(e1[0]) # scalar product with x component of the sensor
        esDoTe1=esDoTe1.at[1].set(e1[1]) # scalar product with y component of the sensor
        esDoTe1=esDoTe1.at[2].set(e1[2]) # scalar product with z component of the sensor

        fp_s1 = np.zeros(3*N)
        fs_s1 = np.zeros(3*N)

        fp_s1=fp_s1.at[0:N].set((jax_bessel.spherical_jv(0,dist1[0:N]*kp) + jax_bessel.spherical_jv(2,dist1[0:N]*kp))*esDoTe1[0] - 3.*jax_bessel.spherical_jv(2,dist1[0:N]*kp)*esDoTes1[0:N]*e1DoTes1[0:N])	
        fs_s1=fs_s1.at[0:N].set((jax_bessel.spherical_jv(0,dist1[0:N]*ks) - 0.5*jax_bessel.spherical_jv(2,dist1[0:N]*ks))*esDoTe1[0] + (3./2)*jax_bessel.spherical_jv(2,dist1[0:N]*ks)*esDoTes1[0:N]*e1DoTes1[0:N])												          																   

        fp_s1=fp_s1.at[N:2*N].set((jax_bessel.spherical_jv(0,dist1[N:2*N]*kp) + jax_bessel.spherical_jv(2,dist1[N:2*N]*kp))*esDoTe1[1] - 3.*jax_bessel.spherical_jv(2,dist1[N:2*N]*kp)*esDoTes1[N:2*N]*e1DoTes1[N:2*N])	
        fs_s1=fs_s1.at[N:2*N].set((jax_bessel.spherical_jv(0,dist1[N:2*N]*ks) - 0.5*jax_bessel.spherical_jv(2,dist1[N:2*N]*ks))*esDoTe1[1] + (3./2)*jax_bessel.spherical_jv(2,dist1[N:2*N]*ks)*esDoTes1[N:2*N]*e1DoTes1[N:2*N])												          																   

        fp_s1=fp_s1.at[2*N:3*N].set((jax_bessel.spherical_jv(0,dist1[2*N:3*N]*kp) + jax_bessel.spherical_jv(2,dist1[2*N:3*N]*kp))*esDoTe1[2] - 3.*jax_bessel.spherical_jv(2,dist1[2*N:3*N]*kp)*esDoTes1[2*N:3*N]*e1DoTes1[2*N:3*N])	
        fs_s1=fs_s1.at[2*N:3*N].set((jax_bessel.spherical_jv(0,dist1[2*N:3*N]*ks) - 0.5*jax_bessel.spherical_jv(2,dist1[2*N:3*N]*ks))*esDoTe1[2] + (3./2)*jax_bessel.spherical_jv(2,dist1[2*N:3*N]*ks)*esDoTes1[2*N:3*N]*e1DoTes1[2*N:3*N])												          																   

        return fp_s1, fs_s1
    
@partial(jax.jit,static_argnums=(2,))
def CSN_END_ch3 (kp, ks, N, x,y,z, SNR, p, e_TestMass, d_TM):

        fp_s1, fs_s1 =  CSN_corr_funct (kp, ks, N, x,y,z, SNR, p, e_TestMass, d_TM)

        #SN:
        Csn = 1/3*(2*p*fp_s1 - (1-p)*fs_s1) #multiplied by (4 pi rho_0 G) S_tot which simplifies with Css and Cnn

        return Csn
    
@partial(jax.jit,static_argnums=(2,))
def CSN_2IN_ch3 (kp, ks, N, x,y,z, SNR, p, e_TestMass1, e_TestMass2, d_TM1, d_TM2):

        fp_s1, fs_s1 =  CSN_corr_funct (kp, ks, N, x,y,z, SNR, p, e_TestMass1, d_TM1)
        fp_s2, fs_s2 =  CSN_corr_funct (kp, ks, N, x,y,z, SNR, p, e_TestMass2, d_TM2)

        #SN:
        Csn = 1/3*(2*p*(fp_s2 - fp_s1) - (1-p)*(fs_s2 - fs_s1)) #multiplied by (4 pi rho_0 G) S_tot which simplifies with Css and Cnn
        return Csn
    
@jax.jit
def CNN_3ch (p):
        #NN:
        Cnn = 1/9*(3*p + 1) #multiplied by (4 pi rho_0 G)^2 S_tot which simplifies with Csn and Css
        return Cnn
    
@jax.jit
def CNN_2IN_3ch (p, kp, ks, e_TestMass1, e_TestMass2, d_TM1, d_TM2):
        #NN:
        e2DoTe1 = np.dot(e_TestMass1,e_TestMass2)
        e21 = d_TM1*e_TestMass1-d_TM2*e_TestMass2
        dist1 = np.linalg.norm((e21))
        e21 = e21/dist1
        e2DoTe21 = np.dot(e_TestMass2,e21)
        e1DoTe21 = np.dot(e_TestMass1,e21)
        
        fp = (jax_bessel.spherical_jv(0,dist1*kp) + jax_bessel.spherical_jv(2,dist1*kp))*e2DoTe1 - 3.*jax_bessel.spherical_jv(2,dist1*kp)*e2DoTe21*e1DoTe21
        fs = (jax_bessel.spherical_jv(0,dist1*ks) - 0.5*jax_bessel.spherical_jv(2,dist1*ks))*e2DoTe1 + (3./2)*jax_bessel.spherical_jv(2,dist1*ks)*e2DoTe21*e1DoTe21
        Cnn = 1/9*(2*(3*p + 1) - 2*(4*p*fp + (1-p)*fs))  #multiplied by (4 pi rho_0 G)^2 S_tot which simplifies with Csn and Css
        return Cnn

@jax.jit
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


@partial(jax.jit,static_argnums=(1,5,))
def Residual (state, N, freq, SNR, p, mirror="all"):
        
        state=np.array(state)
    
        kp = 2*np.pi*freq/6000 #velocity for p-waves 6000 m/s
        ks = 2*np.pi*freq/4000 #velocity for s-waves 4000 m/s
        
        # print(state)
        #coordinate of each seismometer and create matrix of distances between each seismometer
        Res = 1
    
        s = denormalizeCoordinates(state.reshape(N,dim))
        if dim==2:
            phi=s[:,0]*np.pi/180
            theta=s[:,1]*np.pi/180
            x = cavern_radius*np.sin(theta)*np.cos(phi)
            y = cavern_radius*np.sin(theta)*np.sin(phi)
            z = cavern_radius*np.cos(theta)
            if(np.any(cavern_radius**2-x**2-y**2<0) or np.any(cavern_radius**2-x**2-y**2>cavern_radius**2)):
                pass
        else:
            x = s[:,0]
            y = s[:,1]
            z = s[:,2]
        # x = np.array(s[:,0],copy=False,dtype=np.dtype('d'))
        # y = np.array(s[:,1],copy=False,dtype=np.dtype('d'))
        # z = np.array(s[:,2],copy=False,dtype=np.dtype('d'))
    
    
        """************************* correlation between seismometrs calculation: ******************""" 
        Css = CSS_3ch(kp, ks, N, x,y,z, SNR, p)
        """****************** correlation between seismometers and test mass calculation: **************"""
        Csn_in = CSN_2IN_ch3(kp, ks, N, x,y,z, SNR, p, e_TestMass1, e_TestMass2, d_TM_inx, d_TM_iny)
        #Csn_in1 = CSN_END_ch3(kp, ks, N, x,y,z, SNR, p, e_TestMass1, d_TM_inx)
        #Csn_in2 = CSN_END_ch3(kp, ks, N, x,y,z, SNR, p, e_TestMass2, d_TM_iny)
        Csn_end1 = CSN_END_ch3(kp, ks, N, x,y,z, SNR, p, e_TestMass1, d_TM_endx)
        Csn_end2 = CSN_END_ch3(kp, ks, N, x,y,z, SNR, p, e_TestMass2, d_TM_endy)
        """****************** correlation of the test mass: **************"""
        Cnn_END = CNN_3ch(p)
        Cnn_2IN = CNN_2IN_3ch (p, kp, ks, e_TestMass1, e_TestMass2, d_TM_inx, d_TM_iny)

        """ ************* RESIDUAL CALCULATION ***********************"""
        # Csn = [Csn_in] 
        #Csn = [Csn_in1, Csn_in2, Csn_end1, Csn_end2]
        # Cnn = [Cnn_2IN] 
        #Cnn = [Cnn_END, Cnn_END, Cnn_END, Cnn_END]
        Csn = [Csn_in, Csn_end1, Csn_end2]
        Cnn = [Cnn_2IN, Cnn_END, Cnn_END]
        nn = len(Csn)
        Res_v = np.zeros(nn)
        
        #stime=time.time()
        for rr in range(0,nn):
            X = np.linalg.solve(Css, Csn[rr])
            resid = 1-np.dot(Csn[rr],X)/Cnn[rr]
            #resid = 1-np.dot(Csn[rr],np.dot(jax_bessel.differentiable_inv(Css),Csn[rr]))/Cnn[rr]
            #if (resid < 0):
            #    print("NEGATIVE RESIDUAL", resid, "-- rr=", rr)
            #    Css_svd = CSS_svd_3ch(Css)
            #    resid = 1 - np.dot(Csn[rr].conjugate(),np.dot(Css_svd,Csn[rr]))/Cnn[rr]
                # print('residual ', resid)
            Res_v=Res_v.at[rr].set(resid)
            #print("Resiual loop time:",time.time()-stime)
            # Res_Vec = np.sum(Res_v)
        if mirror=="in":
            Res = Res_v[0]
        elif mirror=="in1":
            Res = Res_v[0]
        elif mirror=="in2":
            Res = Res_v[1]
        elif mirror=="end1":
            Res = Res_v[-2]
        elif mirror=="end2":
            Res = Res_v[-1]
        elif mirror=="all":
            Res = np.mean(Res_v)*np.exp(a*np.sum((z/600)**expon))
        else:
            Res = np.max(Res_v)
        return Res

# ************************** SOME ADDITIONAL FUNCTIONS **************************

def getInitialSeismometerPositions(N,bounds=np.array([[-5000,-5000,-300],[5000,5000,300]]),method="random"):
    if method.lower()=="random" or method.lower()=="r":
        pass
    else:
        print("WARNING: unknown seismometer initialization. Using Random instead.")
    if not(bounds.shape[0]==2 and bounds.shape[1]==3):
        print("WARNING: unusual shape for boundaries. Using default.")
        bounds=np.array([[-5000,-5000,-300],[5000,5000,300]])
    output=np.zeros((dim,N))
    for i in range(dim):
        output=output.at[i].set(onp.random.uniform(bounds[0][i],bounds[1][i],N))
    return output.T

def normalizeCoordinates(x,bounds=np.array([[-5000,-5000,-300],[5000,5000,300]])):
    #for i in range(dim):
    #    x=x.at[:,i].set(2*(x[:,i]-bounds[0][i])/(bounds[1][i]-bounds[0][i])-1)
    return x

def denormalizeCoordinates(x,bounds=np.array([[-5000,-5000,-300],[5000,5000,300]])):
    #for i in range(dim):
    #    x=x.at[:,i].set((1+x[:,i])/2*(bounds[1][i]-bounds[0][i])+bounds[0][i])
    return x

def M_to_inv(state,N,freq,SNR,p):
    
    state=np.array(state)

    kp = 2*np.pi*freq/6000 #velocity for p-waves 6000 m/s
    ks = 2*np.pi*freq/4000 #velocity for s-waves 4000 m/s

    s = denormalizeCoordinates(state.reshape(N,dim))
    if dim==2:
        phi=s[:,0]*np.pi/180
        theta=s[:,1]*np.pi/180
        x = cavern_radius*np.sin(theta)*np.cos(phi)
        y = cavern_radius*np.sin(theta)*np.sin(phi)
        z = cavern_radius*np.cos(theta)
    else:
        x = s[:,0]
        y = s[:,1]
        z = s[:,2]
    
    
    return CSS_3ch(kp, ks, N, x,y,z, SNR, p)

"""*********************************************************   MAIN   ******************************************************************"""



#def foo(N=10, ID=0, savename="Results", worker=1, animate=False):
if True:                


        # **************************** DIFFERENTIAL OPTIMIZATION *************************************
        print('starting proc ...')
        starttime=time.time()
        
        precision_cut=1e-3
        maxsteps=10000
        learning_rate=1e-0
        tolerance_steps=500
        if dim==2:
            init_bounds=np.array([[0,0],[360,180]])
        else:
            init_bounds=np.array([[-1000,-1000,-300],[1000,1000,300]])
        
        
        #seismometer_pos=normalizeCoordinates(np.array([[530.0,0.0,0.0]]))
        #seismometer_pos=normalizeCoordinates(np.array([[0., 0., 0.]])) 
        #seismometer_pos=normalizeCoordinates(getInitialSeismometerPositions(N,init_bounds))

        dfdx = jax.jit(jax.grad(Residual),static_argnums=(1,5,))

        prec=1000  
        
        #NoS=4
        #gridscale=1407
        #gridstate=onp.zeros((NoS,NoS,NoS,3))
        #gridstate[:,:,:,0],gridstate[:,:,:,1],gridstate[:,:,:,2]=onp.meshgrid(onp.linspace(-gridscale/2*(NoS-1),gridscale/2*(NoS-1),NoS),onp.linspace(-gridscale/2*(NoS-1),gridscale/2*(NoS-1),NoS),onp.linspace(-gridscale/2*(NoS-1),gridscale/2*(NoS-1),NoS))
        #gridstate.reshape(64,3)
        
        i = 0 #int(argv[1])
        statestring=str(onp.loadtxt(folder+str(N)+tag+"Resultsall"+str(N)+".txt",dtype=str,skiprows=stateline+i*blocksize,max_rows=1,delimiter="ö"))
        pos=np.array(ast.literal_eval(statestring.split("(")[1].split(")")[0]))
        seismometer_pos=normalizeCoordinates(pos)
        
        pos_hist=[seismometer_pos]
        Res_hist=[Residual(pos_hist[-1], N, freq, SNR, p, mirror)]
        nsteps=0  
        
        
        
        
        if whichOptimizer.lower()=="adam":
            optimizer = optax.adam(learning_rate)
            opt_state = optimizer.init(seismometer_pos)
        elif whichOptimizer.lower()=="lbfgs":
            def mirrorRes(state, N, freq, SNR, p, mirror=mirror):
                return Residual(state, N, freq, SNR, p, mirror)
            solver = jaxopt.LBFGS(fun=mirrorRes, history_size=15, maxiter=maxsteps,jit=False)
            #sta = solver.init_state(seismometer_pos, N=N, freq=freq, SNR=SNR, p=p)
            #pa=seismometer_pos
            pa,sta = solver.run(seismometer_pos, N=N, freq=freq, SNR=SNR, p=p)
        elif "scipy" in whichOptimizer.lower():
            seismometer_pos=seismometer_pos.reshape((3*N))
            #optimizerObject = opt.minimize(Residual, seismometer_pos,args=(N,freq,SNR,p,mirror), method="l-bfgs-experimental-do-not-rely-on-this",tol=precision_cut, options={"maxiter":maxsteps})
            optimizerObject = opt.minimize(Residual, seismometer_pos,args=(N,freq,SNR,p,mirror), method="BFGS",tol=precision_cut, options={"maxiter":maxsteps})
            pos_hist.append(optimizerObject.x.reshape(N,dim))
            Res_hist.append(optimizerObject.fun)
       
        mino=1000
        while prec>precision_cut and nsteps<maxsteps:
            if whichOptimizer.lower()=="adam":
                grads = dfdx(pos_hist[-1], N, freq, SNR, p, mirror)
                updates, opt_state = optimizer.update(grads, opt_state)
                pos_hist.append(optax.apply_updates(pos_hist[-1], updates))
                Res_hist.append(Residual(pos_hist[-1], N, freq, SNR, p, mirror))
            elif whichOptimizer.lower()=="lbfgs" and False:
                statime=time.time()
                pa,sta=solver.update(pa, sta, N=N, freq=freq, SNR=SNR, p=p)
                print("time of step"+str(nsteps)+":",time.time()-statime)
                pos_hist.append(sta.s_history[-1])
                Res_hist.append(sta.value)
            nsteps+=1
            if(maxsteps%nsteps==0):
                print(str(nsteps)+"/"+str(maxsteps))
            if nsteps>tolerance_steps:
                if mino>Res_hist[-tolerance_steps-1]:
                    mino=Res_hist[-tolerance_steps-1]
                minnn=onp.min(Res_hist[-tolerance_steps:])
                prec=(mino-minnn)/minnn
            #print(Res_hist[-1])
            
        if whichOptimizer.lower()=="lbfgs":
            pos_hist=list(sta.s_history[:np.argmax(sta.rho_history==0)])
            pos_hist.append(pa)
            Res_hist=[]
            for pos in pos_hist:
                Res_hist.append(Residual(pos,N,freq,SNR,p,mirror))
            print(denormalizeCoordinates(pa.reshape(N,dim)))
            
        fun = min(Res_hist)
        best_x = pos_hist[np.argmax(np.array(Res_hist)==fun)]
        best_x = denormalizeCoordinates(best_x.reshape(N,dim))
        print(fun,best_x)
        
        
        #for i in range(len(pos_hist)):
        #    pos_hist[i]=denormalizeCoordinates(pos_hist[i])
        pos_hist=np.array(pos_hist)
    
        
        
        # ************************** VISUALIZATION **************************
        
        if animate:
            plt.close("all")
            
            plt.figure()
            plt.title("The seismometer positions during optimization")
            plt.xlabel("iteration")
            plt.ylabel("x,y,z")
            for i in range(min(N,3)):
                for j in range(dim): 
                    plt.plot(pos_hist[:,i,j],label="no. {}, {}-pos".format(i,["x","y","z"][j%dim]))
            plt.legend()
            
            
            plt.figure()
            plt.title("Residual during optimization "+"(N={:.0f}, f={:.0f} Hz, SNR={:.0f}, p={:.1f})".format(N,freq,SNR,p))
            plt.xlabel("iteration")
            plt.ylabel("R")
            plt.plot(Res_hist)
            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            if dim==2:
                phi=best_x[:,0]*np.pi/180
                theta=best_x[:,1]*np.pi/180
                bx=cavern_radius*np.sin(theta)*np.cos(phi)
                by=cavern_radius*np.sin(theta)*np.sin(phi)
                bz=cavern_radius*np.cos(theta)
                ax.scatter(bx,by,bz,c='g', marker='o')
            else:
                ax.scatter(best_x[:,0],best_x[:,1],best_x[:,2],c='g', marker='o')
            ax.scatter(d_TM_inx*e_TestMass1[0],d_TM_inx*e_TestMass1[1],d_TM_inx*e_TestMass1[2],c='r', marker='o')
            ax.scatter(d_TM_endx*e_TestMass1[0],d_TM_endx*e_TestMass1[1],d_TM_endx*e_TestMass1[2],c='r', marker='o')
            ax.scatter(d_TM_iny*e_TestMass2[0],d_TM_iny*e_TestMass2[1],d_TM_iny*e_TestMass2[2],c='r', marker='o')
            ax.scatter(d_TM_endy*e_TestMass2[0],d_TM_endy*e_TestMass2[1],d_TM_endy*e_TestMass2[2],c='r', marker='o')
            plt.plot([0,d_TM_endx*e_TestMass1[0]], [0,d_TM_endx*e_TestMass1[1]], '--', c = 'k')
            plt.plot([0,d_TM_endy*e_TestMass2[0]], [0,d_TM_endy*e_TestMass2[1]], '--', c = 'k')
        
            plt.xlabel("x")
            plt.ylabel("y")
            ax.set_zlabel("z")
            ax.set_title("Energie="+str(np.sqrt(fun)))
        
            
        
        # ************************** WRITING FILE **************************    
            
        filename = savename+str(ID)+'.txt'
        f = open(filename,'a+') 


        f.write('\n \n \n## *************'+whichOptimizer+'-bulk-'+ str(ID) + '*************** ##\n \n \n' )
        
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
        
        f.write('initial_pos = np.array('+str(seismometer_pos.tolist())+')\n')

        f.write('optimizer_options = {"learning rate": '+str(learning_rate)+'}\n')

        f.write('Energy = ')
        # remember to take the root, skipped in the optimizing process
        json.dump(float(np.sqrt(fun)),f)
        f.write('\n')
        
        f.write('e2 = ')
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


        aa = '\nFinalState'+str(ID)+' = np.array('
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
        
        print("Finished in",np.round((time.time()-starttime)/60,2),"minutes ("+str(nsteps)+" steps)")
"""
        if "adam" in whichOptimizer.lower():
            return optimizer
        elif "lbfgs" in whichOptimizer.lower():
            return pa,sta
        elif "scipy" in whichOptimizer.lower():
            return optimizerObject
        else:
            return 0
        
#def main(N=15, ID=0, ww=8):

if __name__ == '__main__':
        
        N = 1#int(argv[1]) #number of seismometers
        ID = 0#int(argv[2]) #job identifier 
        savename = "p"+"Results"#str(argv[3]) #name of numerated Savefiles
        ww = 1#int(argv[4]) #workers
        animate=False
        #pool = Pool(processes=6)
        #pool.map(foo, range(30))
    
        #Number of seismometers (At the end it's like having N*3 seismometers since each seismometer is composed by three channels (x,y,z): like having 3 seismometers in N positions)
        optimizer=foo(N, ID, savename, ww, animate)
"""





