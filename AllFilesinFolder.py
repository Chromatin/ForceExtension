# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:52:49 2018

@author: nhermans
"""
import os #filenames
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
#from scipy.special import erf
import Functions as func
import Tools

folder = '2018' #folder with chromosome sequence files (note, do not put other files in this folder)
filenames = os.listdir(folder)
os.chdir( folder )

Select=1 #1 for Selected Data, 0 for all data
Pulling = 1 #1 for only pulling data
DelBreaks =1 # 1 for deleting data after tether breaks
MinForce=2 #only analyze data above this force
steps , stacks = [],[]
plt.close() #close all references to figures still open

for Filename in filenames:
    if Filename[-4:] != '.fit' :
        continue
    Headers,Data = Tools.read_data(Filename)
    LogFile=Tools.read_log(Filename[:-4]+'.log')
    Lc = float(Tools.find_param(LogFile,'L DNA (bp)')) # contour length (bp) #parameters should not be initiliazed here
    p = float(Tools.find_param(LogFile,'p DNA (nm)') )  # persistence length (nm)p DNA (nm)
    S = float(Tools.find_param(LogFile,'S DNA (pN)') ) # stretch modulus (pN) S DNA (pN)
    k = float(Tools.find_param(LogFile,'k folded (pN/nm)') ) # Stiffness of the fiber, in pN/nm => k folded (pN/nm)
    N = float(Tools.find_param(LogFile,'N nuc') )#number of nucleosomes N nuc
    N_tetra = float(Tools.find_param(LogFile,'N unfolded [F0]'))
    NRL = float(Tools.find_param(LogFile,'NRL (bp)') )#NRL (bp
    DNAds =  0.34 # rise per basepair (nm)
    kBT = 4.2 #pn/nm 
    Lmin=Lc-N*NRL+N_tetra*75 # DNA handles in bp
    if Lmin <0: 
        print('bad fit')
        continue
    Lmax=Lc-(N+N_tetra)*75 # Max Z of the "beads on a string" conformation in bp
    Z_fiber = 1 * N #Length of fiber in nm 
    print(Lmin,Lmax,Lc, Filename)
    
    Force = np.array([])
    Time=np.array([])
    Z=np.array([])
    Z_Selected=np.array([])
    
    for idx,item in enumerate(Data):                #Get all the data from the fitfile
        Force=np.append(Force,float(item.split()[Headers.index('F (pN)')]))
        Time=np.append(Time,float(item.split()[Headers.index('t (s)')]))
        Z_Selected=np.append(Z_Selected,float(item.split()[Headers.index('selected z (um)')])*1000)
        Z=np.append(Z,float(item.split()[Headers.index('z (um)')])*1000)
    
    if Select == 1:                                 #If only the selected column is use do this
        ForceSelected = np.delete(Force, np.argwhere(np.isnan(Z_Selected)))
        Z_Selected=np.delete(Z, np.argwhere(np.isnan(Z_Selected)))
    if Select==0:
        ForceSelected=Force
        Z_Selected=Z
    if Pulling ==1: ForceSelected,Z_Selected = func.removerelease(ForceSelected,Z_Selected)
    if DelBreaks ==1: ForceSelected,Z_Selected = func.breaks(ForceSelected,Z_Selected)
    if MinForce > 0: ForceSelected,Z_Selected=func.minforce(ForceSelected,Z_Selected,MinForce)
    
    #Generate FE curves for possible states
    PossibleStates = np.arange(Lmin-200,Lc+50,1) #range to fit 
    dF=0.1 #Used to calculate local stiffness
    ProbSum=np.array([])
    for x in PossibleStates:
        Ratio=func.ratio(Lmin,Lmax,x)
        StateExtension=np.array([func.wlc(ForceSelected,p,S)*x*DNAds + func.hook(ForceSelected,k,10)*Ratio*Z_fiber])
        StateExtension_dF=np.array([func.wlc(ForceSelected+dF,p,S)*x*DNAds + func.hook(ForceSelected+dF,k,10)*Ratio*Z_fiber])
        LocalStiffness = np.subtract(StateExtension_dF,StateExtension)*(kBT) / dF # fix the units of KBT (pN nm -> pN um)
        DeltaZ=np.subtract(Z_Selected,StateExtension)
        std=abs(np.divide(DeltaZ,np.sqrt(LocalStiffness)))
        Pz=np.array([1-func.erfaprox(std)])
        ProbSum=np.append(ProbSum,np.sum(Pz)) 
    PeakInd,Peak=func.findpeaks(ProbSum, 25)
    Peaks = signal.find_peaks_cwt(ProbSum, np.arange(5,30)) #numpy peakfinder, finds too many peaks, not used plot anyway
    States=PossibleStates[PeakInd]
    
    Unwrapsteps=[]
    Stacksteps=[]
    for x in States:
        if x >= Lmax:
            Unwrapsteps.append(x)
        else:
            Stacksteps.append(x)
    Stacksteps=func.state2step(Stacksteps)
    Unwrapsteps=func.state2step(Unwrapsteps)
    if len(Unwrapsteps)>0: steps.extend(Unwrapsteps)
    if len(Stacksteps)>0: stacks.extend(Stacksteps)
    Tools.write_data('AllSteps.txt',Unwrapsteps,Stacksteps)
    
    #plotting
    plt.figure(1)
    plt.xlabel('Extension [nm]')
    plt.ylabel('Force [pN]')
    plt.scatter(Z, Force, color='grey',alpha=0.3, marker='+')
    plt.scatter(Z_Selected,ForceSelected)
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    plt.title(Filename)
    ax1.set_xlabel('time [sec]'), ax2.set_xlabel('Probability [AU]')
    ax1.set_ylabel('Extension [bp]')
    ax1.set_ylim([0,Lc+200])
    ax1.scatter(Time,Z/DNAds)
    ax2.plot(ProbSum,PossibleStates)
    ax2.scatter(Peak,PossibleStates[(PeakInd)])
    ax2.scatter(ProbSum[(Peaks)],PossibleStates[(Peaks)])
    for x in States:
        Ratio=func.ratio(Lmin,Lmax,x)
        Fit=np.array(func.wlc(Force,p,S)*x*DNAds + func.hook(Force,k,10)*Ratio*Z_fiber)
        ax1.plot(Time,Fit/DNAds, linestyle=':')
    plt.savefig(Filename[0:-4]+'_full.png')
    #plt.show()
    
plt.clf()
plt.cla()
plt.hist(steps,  bins = 100, range = [0,200] )
plt.hist(stacks, bins = 100, range = [0,200])
plt.xlabel('stepsize (bp)')
plt.ylabel('Count')
plt.title("Histogram stepsizes in bp")
plt.savefig('hist.png')
plt.show()
