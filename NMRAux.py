import numpy as np
import numpy.typing as npt

from scipy.interpolate import BSpline, make_interp_spline
from scipy.special import binom

from scipy.ndimage import maximum_filter1d, minimum_filter1d, gaussian_filter1d
from scipy.special import expit
from scipy.signal import find_peaks

import nmrglue.analysis.lineshapes1d as ls 
from nmrglue.process.proc_base import ps
from nmrglue.process.proc_autophase import manual_ps

import warnings


ppmRange = 2

nPts = 1024

minCSRange = 0
maxCSRange = minCSRange + ppmRange


CSRange = np.linspace(maxCSRange, minCSRange, nPts)
#Range totale frequenza
# 25 Hz per 128 Pt
# 1600 Hz per 8192 Pt
# 0.1953125 Hz per 1 Pt 
# 1800 Hz per totale
MHZOperative = 1800 / ppmRange #Da reverse engineering

minNarrow, maxNarrow = 0.5, 5
minWide, maxWide = 5,400
minCoupling, maxCoupling = 0.5, 20
minIntensity, maxIntensity = 1, 100
minPhase, maxPhase = -8,8
minDynRange, maxDynRange = 5,10000

# ----

kernelSizes = [128,2048]
def dynamicScaleFiltering(yy):
    yy_final = np.zeros_like(yy)
    #yy_final = yy
    for k in kernelSizes:
        S_max = maximum_filter1d(yy, k)
        S_min = minimum_filter1d(yy, k)
        S_min_g = gaussian_filter1d(S_min, k, mode = "mirror")
        S_max_g = gaussian_filter1d(S_max, k, mode = "mirror")
        yy_final += (yy - S_min_g) / (S_max_g - S_min_g)
    return yy_final / len(kernelSizes)

def peakShrinking(x):
    return expit((x-30) * -1)* 0.6 + 0.4

def fromPeaksToPos(x):
    xx = np.array(x)
    return xx * ppmRange / -nPts + ppmRange

def fromPpmToIndex(x):
    return round(nPts - (x / ppmRange) * nPts)

# ---

def generateRandomSpectrum(seed = None, peakMode = "narrow") -> npt.ArrayLike:
    rng = np.random.default_rng(seed)
    
    assert peakMode in ["narrow", "wide"]
    
    freq = MHZOperative
    hztoppm = lambda x: x / (freq)
    #Let's fix the max ppm 
    minppm = minCSRange
    maxppm = maxCSRange
    
    peakAmt = rng.integers(8) + 1
    
    peaksPpm = rng.uniform(minppm, maxppm, peakAmt)
    multiplicity = rng.integers(5, size = peakAmt) + 1
    width = rng.uniform(hztoppm(minNarrow), hztoppm(maxNarrow), peakAmt)
    intensity = rng.uniform(minIntensity, maxIntensity, peakAmt)
    coupling = rng.uniform(hztoppm(minCoupling), hztoppm(maxCoupling), peakAmt)
    ratios = rng.uniform(0,1, peakAmt)
    
    
    yy = np.zeros_like(CSRange)
    
    peaks = []
    
    for i in range(peakAmt):
        #print(peaksPpm[i])
        #Offset and peaks
        totLength = coupling[i] * multiplicity[i] / 2
        pks = np.linspace(peaksPpm[i] - totLength, peaksPpm[i] + totLength, multiplicity[i])
        
        maxInt = binom(multiplicity[i]-1, (multiplicity[i])//2)
    
        for j,pk in enumerate(pks):
            
            if fromPpmToIndex(pk) < nPts and peakMode == "narrow":
                peaks.append(fromPpmToIndex(pk))            
                        
            multiplier = binom(multiplicity[i]-1, j) / maxInt
            yy += intensity[i] * ls.sim_pvoigt_fwhm(CSRange, pk, width[i], ratios[i]) * multiplier
            
    #BIGPEAKS
    
    bigPeakAmt = rng.integers(3)
    bigPeaksPpm = rng.uniform(minppm, maxppm, bigPeakAmt)
    bigIntensity = rng.uniform(minIntensity, maxIntensity, bigPeakAmt) * 0.1
    bigWidth = rng.uniform(hztoppm(minWide), hztoppm(maxWide), bigPeakAmt)
    bigRatios = rng.uniform(0,1, bigPeakAmt)
    
    for i in range(bigPeakAmt):
        bigPk = bigPeaksPpm[i]
        if fromPpmToIndex(bigPk) < nPts and peakMode == "wide":
            peaks.append(fromPpmToIndex(bigPk))   
        
        yy += bigIntensity[i] * ls.sim_pvoigt_fwhm(CSRange, bigPk, bigWidth[i], bigRatios[i])
    
    #DISTORTION
    
    maxNoise = rng.uniform(1e-4, 1e-2) * np.max(intensity)
    
    if rng.uniform() <= 0.2:
        maxDist = rng.uniform(0.5, 2.5) * maxNoise        
        xSpline = np.linspace(minCSRange, maxCSRange, 10)
        ySpline = rng.normal(scale = maxDist, size = 10)
        spl = make_interp_spline(xSpline, ySpline)
        yy += spl(CSRange)
    
    #Solvent Peak
    if rng.uniform() <= 0.05:
        solvPeaksPpm = rng.uniform(minppm, maxppm)
        solvWidth = 0.005
        solvRatio = rng.uniform(0,1)
        solvIntensity = rng.uniform(0,1500)
        yy += np.max(intensity) * solvIntensity * ls.sim_pvoigt_fwhm(CSRange, pk, solvWidth, solvRatio)
        
    #PHASE
        
    phase = rng.uniform(minPhase, maxPhase)
    with warnings.catch_warnings(action="ignore"):
        yy = ps(yy, p1 = phase)
    
    #NOISE
    noise = rng.normal(scale = maxNoise, size = nPts)
    SNR = yy/(maxNoise/2)      
    yyReg = yy * peakShrinking(SNR)
    
    yyFiltered = dynamicScaleFiltering(yy)  
    
    '''
    
        
    peaks, properties = find_peaks(yy, prominence=10, width=0.1)
    wl =  fromPeaksToPos(properties["left_ips"])
    wr =  fromPeaksToPos(properties["right_ips"])
    
    xPks = CSRange[peaks]
    yPks = yyPP[peaks]
    wPks = wr - wl
        
    retStats.append({"xPks": xPks, "yPks":yPks, "wPks": wPks, "peaks": peaks})
    '''

    
    return {"pure": yy, "true": yy + noise, "filtered": yyFiltered, "reg": yyReg}, peaks