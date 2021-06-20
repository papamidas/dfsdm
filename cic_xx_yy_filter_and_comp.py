#-*- coding: utf-8 -*-
"""
created on June 19, 2021

STM32H7 DFSDM filter simulation

@author: papamidas DM1CR

This script shows the filter curve of one CIC filter with a high downsampling
factor and realizable with the DFSDM unit inside of a STM32 microcontroller
vs. the filter curve of two cascaded CIC filters with approximately the same
downsampling factor (see www.dm1cr.de).
A DFSDM unit of a STM32 microcontroller provides four filters, so lowpass
filtering of an IQ signal with two cascaded CIC filters for each I and Q
signal is feasible.

See "en.dfsdm_tutorial.xlsx", issued by ST.   
"""

import numpy as np
from scipy.signal import firwin2, upfirdn
import matplotlib.pyplot as plt
Nfft = 256

##################### Single CIC-Filter
FORD_single = 2 # SINC-Filter-Order of single filter
sample_rate_single=6070e3/2.25
FOSR_single= 128 # single SINC-Filter oversampling ratio
h_single = np.ones(FOSR_single)
# plot the frequency response:
# take FFT and magnitude
H_single = np.abs(np.fft.fft(h_single, Nfft*FOSR_single)**FORD_single)
# make 0 Hz in the center
H_single = np.fft.fftshift(H_single)
# x axis:
w = np.linspace(-sample_rate_single/2, sample_rate_single/2, len(H_single))
# log scaling:
HdB_single = 20 * np.log10(H_single+1e-3) # add min level to avoid log(0)
HdBmax_single = HdB_single[np.argmax(HdB_single)] # search for max gain
HdB_single -= HdBmax_single # and set max gain to 0 dB
plt.plot(w, HdB_single, label = f'single CIC, FOSR={FOSR_single}' +
         f', FORD={FORD_single}')

##################### CIC-Filter #1
FORD_first = 4 # SINC-Filter-Order of first filter
sample_rate=6070e3/2.25
FOSR_first = 15 # first SINC-Filter oversampling ratio
h_first = np.ones(FOSR_first)
# plot the frequency response:
 # take FFT and magnitude
H_first = np.abs(np.fft.fft(h_first, Nfft*FOSR_single)**FORD_first)
 # make 0 Hz in the center
H_first = np.fft.fftshift(H_first)
# log scaling:
HdB_first = 20 * np.log10(H_first+1e-3)  # add min level to avoid log(0)
HdBmax_first = HdB_first[np.argmax(HdB_first)] # search for max  gain
HdB_first -= HdBmax_first # and set max gain to 0 dB
plt.plot(w, HdB_first, label = f'cascaded CIC #1, FOSR={FOSR_first}' +
         f', FORD={FORD_first}')

##################### CIC-Filter #2
FORD_second = 5 # SINC-Filter-Order of second filter
sample_rate_second = sample_rate/FOSR_first # input sample rate of 2nd filter
FOSR_second = 8 # second SINC-Filter oversampling ratio
h_second = np.ones(FOSR_second)
h_second = upfirdn([1], h_second, up=FOSR_first) # second filter upsampled
# plot the frequency response
 # take FFT and magnitude
H_second = np.abs(np.fft.fft(h_second, Nfft*FOSR_single)**FORD_second)
 # make 0 Hz in the center
H_second = np.fft.fftshift(H_second)
# log scaling:
HdB_second = 20 * np.log10(H_second+1e-3)
HdBmax_second = HdB_second[np.argmax(HdB_second)]
HdB_second -= HdBmax_second
plt.plot(w, HdB_second, label = f'cascaded CIC #2, FOSR={FOSR_second}' +
         f', FORD={FORD_second}')

# plot combined action of first and second filter:
plt.plot(w, HdB_first+HdB_second, label = 'cascaded CIC#1 -> CIC#2')


# CIC compensation filter design:

#----------------------------------------
# A good practice is to choose the pass band edge to be less than a
# quarter of the first null on the low frequency scale fS/R.

# compensation calculation taken from Altera Appnote AN455
# and translated from MATLAB to Python:
###### CIC filter parameters ######
R = FOSR_second                ## Decimation factor
M = 1                          ## Differential delay
N = FORD_second                ## Number of stages
Fs = sample_rate_second        ## (High) Sampling freq in Hz before decimation
Fc = Fs/R/4                    ## Pass band edge in Hz

####### fir2.m parameters ######
L = 30                         ## Filter order must be even
Fo = R*Fc/Fs*0.6                   ## Normalized Cutoff freq 0<Fo<=0.5/M
#Fo = 0.49/M                    ## use Fo=0.5 if you don't care responses are
                                ## outside the pass band

####### CIC Compensator Design using firwin2 ######
p=2e3                          ## Granularity
s=0.25/p                       ## Step size
fp = np.arange(0, Fo+s, s)     ## Pass band frequency samples
fs = np.arange(Fo+s, 0.5+s, s) ## Stop band frequency samples
f = 2 * np.concatenate((fp,fs))## Normalized frequency samples 0<=f<=1
Mp = np.ones(len(fp))          ## Pass band response Mp(1)=1
Mpnum = M * R * np.sin(np.pi * fp/R)
Mpden = np.sin(np.pi * M * fp)
Mp[1:] = np.abs( Mpnum[1:]/Mpden[1:])**N
Mf = np.concatenate((Mp, np.zeros(len(fs))))
f[-1] = 1
hcomp = firwin2(L+1, f, Mf)        ## Filter length L+1
hcomp = hcomp/np.max(hcomp)        ## Floating point coefficients

#up-filtering of the filter in order to get response over whole bandwidth 
hcomp_up = upfirdn([1], hcomp, up=FOSR_single)
# take FFT and magnitude:
Hcomp_up = np.abs(np.fft.fft(hcomp_up, Nfft*FOSR_single))
Hcomp_up = np.fft.fftshift(Hcomp_up) # make 0 Hz in the center
HdBcomp_up = 20 * np.log10(Hcomp_up+1e-3)
HdBcomp_up_max = HdBcomp_up[np.argmax(HdBcomp_up)]
HdBcomp_up -= HdBcomp_up_max
plt.plot(w, HdBcomp_up, label='CIC compensation filter')

plt.plot(w, HdB_first+HdB_second+HdBcomp_up,
         label = 'cascaded CIC#1 -> CIC#2 -> CIC comp. filter')

plt.grid()
plt.legend()
plt.show()
