# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 11:22:15 2017

@author: noort
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import fileio as fio
from lmfit import Minimizer, Parameters, report_fit
import collections

kT = 4.1 #(pN nm)
L0 = 0.34 #(nm/bp)

def fiber(f, par = None):
    """  
    Calculate the extension of a nucleosome embeded in a folded chromatin fiber.
    
    Parameters
    ----------
    par : parameter
    
    Returns
    -------
    z : nparray
        Extension (nm) of the fiber per numcleosome.
    work: nparry
        Work (kT) associated with stretchingthe fiber
        
    References
    ----------
    .. [1] Quantitative analysis of single-molecule force spectroscopy on folded
       chromatin fibers, He Meng,  Kurt Andresen and John van Noort
       Nucleic Acids Research, Volume 43, Issue 7, 20 April 2015, Pages 3578–3590
       https://doi-org.ezproxy.leidenuniv.nl:2443/10.1093/nar/gkv215

    """
    if par == None:
        par = Parameters()
        par.add('NLD_nm', value=1.7)
        par.add('k_pN_nm',value=0.3)
    p = par.valuesdict()
    
    z = f/p['k_pN_nm'] + p['NLD_nm']
    z = np.clip(z, 0, 15)
    work = - (f**2)/ (2*p['k_pN_nm']) - f * p['NLD_nm']
    return z, work/kT

def WLC(f, par = None):
    if par == None:
        par = Parameters()
        par.add('L_bp',   value=3e3)
        par.add('P_nm',   value=50)
        par.add('S_pN',   value=1e3)
    p = par.valuesdict()
    
    z = 1- 0.5* np.sqrt(kT/(f*p['P_nm'])) + f/p['S_pN']
    z = p['L_bp'] * z * L0
    work = f - np.sqrt(f * kT/p['P_nm']) +f**2/(2*p['S_pN'])
    work = p['L_bp'] * work
    return z, work/kT

def _fiberWLC_gen_distribution(N4, N8, degeneracy = 0, G1 = 0, G2 = 0, G3 = 0):
    N =[]
    D = []
    G = []
    for n1 in range(N8+1):
        N.append([N8 - n1, n1, N4, 0])
        if degeneracy == 1:          
            D.append(sp.special.binom(N8, n1))
        else:
            D.append(sp.special.binom(N8, n1))
        G.append(n1*G1)
    for n2 in range(1, N8+1):
        N.append([0, N8 - n2, n2+N4, 0])
        D.append(sp.special.binom(N8, n2))  
        G.append((N8 - n2)*G1 + n2*(G1+G2))
    for n3 in range(1, N4+N8+1):
        N.append([0, 0, N4+N8 - n3, n3])
        D.append(sp.special.binom(N4+N8, n3))  
        G.append(N8*(G1 + G2) + n3* G3)
        
    return np.asarray(N), np.asarray(D), np.asarray(G)*kT

def OLD_fiberWLC_gen_distribution(N4, N8, degeneracy = 0, G1 = 0, G2 = 0, G3 = 0):
    N =[]
    D = []
    G = []
    for n0 in range(N8+1):
        for n1 in range(N8+1):
            for n2 in range(N4+N8+1):
                for n3 in range(N4+N8+1):
                    if ((n0 + n1 <= N8) &
                        (n0+n1+n2+n3 == N4+N8 ) &
                        (n2+n3 >= N4)):
                            N.append([n0, n1, n2, n3])
                            d = sp.special.binom(n0 + n1 + n2 +n3, n1)
                            d *= sp.special.binom(n0 + n1 + n2 +n3, n2)
                            d *= sp.special.binom(n0 + n1 + n2 +n3, n3)                      
                            D.append(d)
                            g = n1*G1 + n2*(G1+G2) + (n3-N4)*(G1+G2) + n3*G3
                            G.append(g*kT)
    return np.asarray(N[::-1]), np.asarray(D[::-1]), np.asarray(G[::-1])

def fiberWLC(f, par = None):
    if par == None:
        par = Parameters()
        par.add('L_bp',       value=3e3)
        par.add('P_nm',       value=50)
        par.add('S_pN',       value=1e3)
        par.add('NRL_bp',     value=197)
        par.add('NLD_nm',     value=1.7)
        par.add('k_pN_nm',    value=0.3)
        par.add('Lwrap1_bp',  value=80)
        par.add('Lwrap2_bp',  value=60)
        par.add('G1_kT',      value=5)
        par.add('G2_kT',      value=20)
        par.add('G3_kT',      value=40)
        par.add('N4',         value=0)
        par.add('N8',         value=15)
        par.add('degeneracy', value=1)
        p = par.valuesdict()
    else:
        p = par.valuesdict()

    # generate per state: distribution of conformations, degeneracy and rupture energy
    Ni, Di, G0i = _fiberWLC_gen_distribution(
            p['N4'], p['N8'], 
            G1 = p['G1_kT'], G2 = p['G2_kT'], G3 = p['G3_kT'],
            degeneracy=p['degeneracy'])
#    for n, g, d in zip(Ni, G0i, Di):
#        print 'ni: ',n, 'Grupt:', g/kT,'(kT)   degeneracy:', d 

    # number of wrapped bps per nucleosome conformation
    Li =[p['NRL_bp'], p['Lwrap1_bp'], p['Lwrap2_bp'], 0] 

    # total number of free bps per state
    Lfree_i = p['L_bp']- np.sum(np.multiply(Li, Ni), axis = 1)
    Lfree_i = np.clip(Lfree_i, 0, p['L_bp'])

    # extension and work per nucleosome and per bp
    z_fiber, w_fiber = fiber(f, par = par)
    z_DNA, w_DNA = WLC(f, par = par)
    z_DNA = z_DNA/p['L_bp']
    w_DNA = w_DNA/p['L_bp']

    # iterate force and caculate partition function
    z_fiberWLC=[]
    for z_nuc, z_bp, w_nuc, w_bp in zip(z_fiber, z_DNA, w_fiber, w_DNA):
        z_state = Ni[:,0]*z_nuc + Lfree_i*z_bp
        g_state = G0i -(Ni[:,0]*w_nuc + Lfree_i*w_bp)
        g_state -= np.min(g_state)

        Z = Di*np.exp(-g_state)/np.sum(Di*np.exp(-g_state))
    
        z_fiberWLC.append(np.sum(z_state*Z))

    return z_fiberWLC


def main():
    f = np.logspace(-1.67, 1.3, 10000)

    p = Parameters()
    p.add('L_bp',   value=6e3,    min = 0,       max = 1e4,     vary = False)
    p.add('P_nm',   value=50,     min = 1,       max = 100,     vary = False)
    p.add('S_pN',   value=900,    min = 100,     max = 2000,    vary = False)
    p.add('NRL_bp', value=197,    min = 147,     max = 300,     vary = False)
    p.add('NLD_nm', value=1.7,    min = 1,       max = 10,      vary = False)
    p.add('k_pN_nm',value=1.3,    min = 0.05,    max = 5,       vary = False)
    p.add('Lwrap1_bp', value=100, min = 1,       max = 147,     vary = False)
    p.add('Lwrap2_bp', value=80,  min = 1,       max = 147,     vary = False)
    p.add('G1_kT',  value=18,     min = 1,       max = 40,      vary = False)
    p.add('G2_kT',  value=4.1,    min = 1,       max = 40,      vary = False)
    p.add('G3_kT',  value=80,     min = 1,       max = 400,     vary = False)
    p.add('N8',     value= 15 ,   min = 0,       max = 100,     vary = False)
    p.add('N4',     value=0,      min = 0,       max = 100,     vary = False)
    p.add('degeneracy', value=1,  min = 0,       max = 1,       vary = False)
       
    z = fiberWLC(f, par = p)
    par = p.valuesdict()
    N, D, G = _fiberWLC_gen_distribution(par['N4'], par['N8'], degeneracy = 1,
             G1 = par['G1_kT'], G2 = par['G2_kT'], G3 = par['G3_kT'] )
    for n, d, g  in zip(N, D, G):
        print n,  '\t', int(d), '\t', g
#    i = range(len(D))
    plt.plot(z, f)
    plt.show()
    
    return

#    fio.write_tdms('c:\\temp\\test.tdms','group1','f_pN', f)
#    fio.write_tdms('c:\\temp\\test.tdms','group1','z_nm', z, p.valuesdict())
#    fio.write_tdms('c:\\temp\\test.tdms','group1','G_kT', G, p.valuesdict())

#    N, D = fiberWLC_gen_distribution(2, 8, degeneracy = 1)
#    i = range(len(D))
#    i = np.asarray(i)
#    i = i[::-1]
#    for n in N:
#    plt.plot(i,D)
    
    plt.plot(z/1000, f)
    plt. ylim([0,20])
    plt. xlim([0,2.5])
    plt.show()
    
    
    return


if __name__ == "__main__":
    # execute only if run as a script
    main()