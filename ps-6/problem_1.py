#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 23:35:25 2023

@author: belper
"""

import astropy.io.fits as fits
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import time


#Loading data
hdu_list=fits.open('specgrid.fits')
logwave = hdu_list['LOGWAVE'].data
flux=hdu_list['FLUX'].data

#plot
for i in range(5):
    plt.plot(logwave, flux[i,:],label=f'galaxy {i}')
plt.xlabel('Wavelength (A) (logscale)')
plt.ylabel('Flux (10^-17 erg/(sAcm^2))')
plt.legend()
plt.savefig('part_a')
plt.show()


for i in range(5):
    plt.plot(logwave, flux[i,:],label=f'galaxy {i}')

plt.axvline(x=np.log10(6563),c='k',alpha=.5)
plt.axvline(x=np.log10(4861),c='k',alpha=.5)
plt.axvline(x=np.log10(4340),c='k',alpha=.5)
plt.axvline(x=np.log10(3970),c='k',alpha=.5)
plt.xlabel('Wavelength (A) (logscale)')
plt.ylabel('Flux (10^-17 erg/(sAcm^2))')
plt.legend()
plt.savefig('part_a_hydrogen')
plt.show()

N_gal=np.shape(flux)[0]
#b
flux_norms=np.sum(flux,axis=1)
nflux=(flux.T/flux_norms).T

#c
flux_means=np.mean(nflux,axis=1)
nmflux=(nflux.T-flux_means).T
for i in range(5):
    plt.plot(logwave, nmflux[i,:],label=f'galaxy {i}')
plt.xlabel('Wavelength (A) (logscale)')
plt.ylabel('Normalized Flux')
plt.legend()
plt.savefig('part_bc')
plt.show()

#d
def sorted_eigs(r, return_eigvalues = False):
    """
    Calculate the eigenvectors and eigenvalues of the correlation matrix of r
    -----------------------------------------------------
    """
    corr=r.T@r
    eigs=np.linalg.eig(corr) #calculate eigenvectors and values of original 
    arg=np.argsort(eigs[0])[::-1] #get indices for sorted eigenvalues
    eigvec=eigs[1][:,arg] #sort eigenvectors
    eig = eigs[0][arg] # sort eigenvalues
    if return_eigvalues == True:
        return eig, eigvec
    else:
        return eigvec

eigval,eigvec=sorted_eigs(nmflux,return_eigvalues=True)



U,S,VT=la.svd(nmflux,full_matrices=True)
eigvec_svd=VT.T
eigval_svd=S**2
svd_sort = np.argsort(eigval_svd)[::-1]
eigvec_svd = eigvec_svd[:,svd_sort]
eigval_svd = eigval_svd[svd_sort]

for i in range(5):
    plt.plot(logwave, eigvec[i],label=f'eigvec {i}')
plt.xlabel('Wavelength (A) (logscale)')
plt.ylabel('Eigenvector Amplitudes')
plt.legend()
plt.savefig('first_eigvecs')
plt.show() 

[plt.plot(eigvec_svd[:,i], eigvec[:,i], 'o')for i in range(len(eigvec))]
plt.plot(np.linspace(-0.2, 0.2), np.linspace(-0.2, 0.2))
plt.xlabel('SVD eigenvector')
plt.ylabel('covariance eigenvector')
plt.savefig('eigenvec_plot')
plt.show()

#g
def PCA(l, r, project = True):
    """
    Perform PCA dimensionality reduction
    --------------------------------------------------------------------------------------
    """
    eigvector = sorted_eigs(r)
    eigvec=eigvector[:,:l] #sort eigenvectors, only keep l
    reduced_wavelength_data= np.dot(eigvec.T,r.T) #np.dot(eigvec.T, np.dot(eigvec,r.T))
    if project == False:
        return reduced_wavelength_data.T # get the reduced wavelength weights
    else: 
        return reduced_wavelength_data.T, np.dot(eigvec, reduced_wavelength_data).T # multiply eigenvectors by 
                                                        # weights to get approximate spectrum
                                                     
plt.plot(logwave,nmflux[1],label="data")
plt.plot(logwave,PCA(5,nmflux)[1][1],label="approximation")
plt.xlabel('Wavelength (A) (logscale)')
plt.ylabel('Normalized Flux')
plt.legend()
plt.savefig("5-component_approx")
plt.show()

#h,i
def PCA_fast(l, r, eigvec):
    eigvec=eigvec[:,:l] #sort eigenvectors, only keep l
    reduced_wavelength_data= np.dot(eigvec.T,r.T) #np.dot(eigvec.T, np.dot(eigvec,r.T))
    return reduced_wavelength_data.T, np.dot(eigvec, reduced_wavelength_data).T # multiply eigenvectors by 
                                                        # weights to get approximate spectrum
                                                        
def weights(l,r,eigvec):
    eigvec=eigvec[:,l]
    reduced_wavelength_data=np.dot(eigvec.T,r.T)
    return reduced_wavelength_data

c_0=weights(0,nmflux, eigvec)
c_1=weights(1,nmflux, eigvec)
c_2=weights(2,nmflux,eigvec)

plt.scatter(c_0, c_1)
plt.xlabel('c_0')
plt.ylabel('c_1')
plt.savefig('c_0c_1')
plt.show()

plt.scatter(c_0, c_2)
plt.xlabel('c_0')
plt.ylabel('c_2')
plt.savefig('c_0c_2')
plt.ylim(-.001,.001)
plt.show()

N_c=list(range(20))
sq_res=[]
for i in range(20):
    proj=PCA_fast(i+1, nmflux, eigvec)[1]
    res=(nmflux-proj)
    sq_res.append(np.sum(res**2))

plt.plot(N_c,sq_res)
plt.xlabel("N_c")
plt.ylabel("Square Residuals")
plt.savefig('residuals_vs_points')
plt.show()
    

    










