from scipy.signal import blackmanharris, triang
from scipy.fftpack import ifft, fftshift
import math
# function to call the main analysis/synthesis functions in software/models/sineModel.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window,resample
import os, sys
import utilFunctions as UF
import sineModel as SM

from scipy.optimize import minimize as mn
from sklearn.metrics import mean_squared_error as met
from utilFunctions import peakDetection as peak
from scipy.signal import resample
import dftModel as DFT
from sineModel import sineTracking, cleaningSineTracks


# Save Overtone Data
def write(f,m,p,sound='str',num=0):
    if sound!='str' and sound!='syn' and sound!='stiff':
        return('Incorrect value for sound')
    if sound=='str':
        os.chdir('/home/tgoodall/sms-tools/software/models/Overtone_Arrays/Strings')
    elif sound=='syn':
        os.chdir('/home/tgoodall/sms-tools/software/models/Overtone_Arrays/Synths')
    np.save('P'+sound+str(num),p)
    np.save('F'+sound+str(num),f)
    np.save('M'+sound+str(num),m)
        
        
def write_stif(stiff,num):
    os.chdir('/home/tgoodall/sms-tools/software/models/Overtone_Arrays/Stiff')
    np.save('stiff '+ str(num),stiff)
    
# Read overtone data
def read(sound='str',num=0):
    if sound!='str' and sound!='syn' and sound!='stiff':
        return('Incorrect value for sound')
    if sound=='str':
        d = ('/home/tgoodall/sms-tools/software/models/Overtone_Arrays/Strings/')
        phase = np.load(d+'P'+sound+str(num)+'.npy')
        freq = np.load(d+'F'+sound+str(num)+'.npy')
        mag = np.load(d+'M'+sound+str(num)+'.npy')
        return freq,mag,phase
    elif sound=='syn':
        d = ('/home/tgoodall/sms-tools/software/models/Overtone_Arrays/Synths/')
        phase = np.load(d+'P'+sound+str(num)+'.npy')
        freq = np.load(d+'F'+sound+str(num)+'.npy')
        mag = np.load(d+'M'+sound+str(num)+'.npy')
        return freq,mag,phase
    elif sound=='stiff':
        d = ('/home/tgoodall/sms-tools/software/models/Overtone_Arrays/Stiff/')
        stiff = np.load(d+'stiff '+str(num)+'.npy')
        return stiff

    
def synth(f,m,p,name,N=4096,H=128,fs=44100):
    hN = N//2                                               # half of FFT size for synthesis
    L = f.shape[0]                                      # number of frames
    pout = 0                                                # initialize output sound pointer         
    ysize = H*(L+3)                                         # output sound size
    y = np.zeros(ysize)                                     # initialize output array
    sw = np.zeros(N)                                        # initialize synthesis window
    ow = triang(2*H)                                        # triangular window
    sw[hN-H:hN+H] = ow                                      # add triangular window
    bh = blackmanharris(N)                                  # blackmanharris window
    bh = bh / sum(bh)                                       # normalized blackmanharris window
    sw[hN-H:hN+H] = sw[hN-H:hN+H]/bh[hN-H:hN+H]             # normalized synthesis window
    lastytfreq = f[0,:]                                 # initialize synthesis frequencies
    ytphase = 2*np.pi*np.random.rand(f[0,:].size)       # initialize synthesis phases 
    err = int(163200/H)                                      # Frame of error
    for l in range(L):                                       
        if pout > y.shape[0]-N:                           # break at penultimate frame
            break
        if (p.size > 0):                                 # if no phases generate them
            ytphase = p[l,:] 
        else:
            ytphase += (np.pi*(lastytfreq+f[l,:])/fs)*H     # propagate phases
        Y = UF.genSpecSines(f[l,:], m[l,:], ytphase, N, fs)  # generate sines in the spectrum   
        Y[np.isnan(Y)]=0
        lastytfreq = f[l,:]                               # save frequency for phase propagation
        yw = np.real(fftshift(ifft(Y)))                       # compute inverse FFT
        y[pout:pout+N] += sw*yw                               # overlap-add and apply a synthesis window
        pout += H                                             # advance sound pointer
    y = np.delete(y, range(hN))                             # delete half of first window
    y = np.delete(y, range(y.size-hN, y.size))              # delete half of the last window
    os.chdir('/home/tgoodall/sms-tools/software/models_interface')
    outputFile = 'output_sounds/' +name+'.wav'
    UF.wavwrite(y, fs, outputFile)
    return y



# synthesize from overtone parameters
def synth_rs(xr,f,m,p,name,N=4096,H=128,fs=44100):
    hN = N//2                                               # half of FFT size for synthesis
    L = f.shape[0]                                      # number of frames
    pout = 0                                                # initialize output sound pointer         
    ysize = H*(L+3)                                         # output sound size
    y = np.zeros(ysize)                                     # initialize output array
    sw = np.zeros(N)                                        # initialize synthesis window
    ow = triang(2*H)                                        # triangular window
    sw[hN-H:hN+H] = ow                                      # add triangular window
    bh = blackmanharris(N)                                  # blackmanharris window
    bh = bh / sum(bh)                                       # normalized blackmanharris window
    sw[hN-H:hN+H] = sw[hN-H:hN+H]/bh[hN-H:hN+H]             # normalized synthesis window
    lastytfreq = f[0,:]                                 # initialize synthesis frequencies
    ytphase = 2*np.pi*np.random.rand(f[0,:].size)       # initialize synthesis phases 
    err = int(163200/H)                                      # Frame of error
    for l in range(L):                                       
        if pout > y.shape[0]-N:                           # break at penultimate frame
            break
        if (p.size > 0):                                 # if no phases generate them
            ytphase = p[l,:] 
        else:
            ytphase += (np.pi*(lastytfreq+f[l,:])/fs)*H     # propagate phases
        Y = UF.genSpecSines(f[l,:], m[l,:], ytphase, N, fs)  # generate sines in the spectrum   
        Y[np.isnan(Y)]=0
        lastytfreq = f[l,:]                               # save frequency for phase propagation
        yw = np.real(fftshift(ifft(Y)))                       # compute inverse FFT
        y[pout:pout+N] += sw*yw                               # overlap-add and apply a synthesis window
        pout += H                                             # advance sound pointer
    y = np.delete(y, range(hN))                             # delete half of first window
    y = np.delete(y, range(y.size-hN, y.size))              # delete half of the last window
    Sy = y.shape[0]
    Sxr = xr.shape[0]
    if Sy>Sxr:
        y = y[Sxr-Sy:]
    elif Sy<Sxr:
        xr = xr[Sxr-Sy:]
    yrs = y + xr
    os.chdir('/home/tgoodall/sms-tools/software/models_interface')
    outputFile = 'output_sounds/' +name+'.wav'
    UF.wavwrite(yrs, fs, outputFile)
    return yrs



def rvals(m,f,init=0):
    iw = m[init,:]                      # initial waveform
    numS = m.shape[1]
    frames = m.shape[0]
    for i in range(1,numS):
        mag = m[:,i]                       # magnitude array for overtone i
        if all(np.isnan(mag)):
            pass
        else:
            freq = f[:,i]                      # freq array for overtone i
            nans = np.isnan(mag)               # nans index for mag
            peak = np.where(mag==max(mag[np.invert(nans)]))[0][0]       # index of highest real value
            nind = np.where(nans==True)[0]            # index values for nans
            if nind.shape[0]!=0:
                nind = nind[nind>peak]                    # indexes for nans after peak
                fin = min(nind)                           # last frame index
                m[fin:,i] = np.nan                      # set all following values to nan
                f[fin:,i] = np.nan                      # set all following values to nan
            else:
                pass                      # real mag values
                        
    return m,f


    
def Ifil (N,frame,smooth):
    for i in range(1,frame.shape[0]):
        if np.isnan(frame[i]):
            frame[i] = frame[i-1]
    frame = frame-frame[0]
    mxsmooth1 = resample(np.maximum(-200,frame),round(frame.size*smooth))
    mxsmooth2 = resample(mxsmooth1,int(N/2))

    return (mxsmooth2)

def H_op_full(f):
    inharm = 0.0146255468054916
    frames = f.shape[0]
    stiff = ([])
    for i in range(frames):
        def H_op(inharm,frame=i,f=f):
            tones = f[frame,1:]
            rind = np.where(np.isnan(tones)==False)[0]
            rtones = tones[rind]
            f0 = f[frame,0]
            sim = ([])
            for i in range(rind.shape[0]):
                k = rind[i]+2
                harm = rtones[i]
                val = ofreq(f0,k=k,inharm=inharm)
                sim = np.append(sim,val)
            return met(sim,rtones)
        sol = mn(H_op,inharm)
        stiff = np.append(stiff,sol.x[0])
    return stiff

def ofreq(f0,k,inharm):
    ovf = k*f0*(1+inharm+inharm**2+((k**2+np.pi**2)/8)*(inharm**2))
    return ovf

def Hdiff(inharm,frame,f,k):
    f0 = f[frame,0]
    ov = f[frame,(k-1)]
    ovf = ofreq(f0,k,inharm)
    dif = abs(ovf-ov)
    return dif

def H_op(inharm,frame,f):
    tones = f[frame,1:]
    rind = np.where(np.isnan(tones)==False)[0]
    rtones = tones[rind]
    f0 = f[frame,0]
    sim = ([])
    for i in range(rind.shape[0]):
        k = rind[i]+2
        harm = rtones[i]
        val = ofreq(f0,k=k,inharm=inharm)
        sim = np.append(sim,val)
    return met(sim,rtones)

def onset(ffts):
    
    frames = ffts.shape[0]
    flux = ([])
    mags = 10**(ffts/20)
    for i in range(1,frames):
        flux = np.append(flux,sum(mags[i,:]-mags[i-1,:]))
    ot = 0.02 # onset threshold
    iflux = flux[flux>ot] # indexed flux values from threshold
    init = peak(iflux,ot) # Onset index
    ind = np.where(iflux[init[0]]==flux)
    return ind

def Lindft(x, w, N):
    """
    Analysis of a signal using the discrete Fourier transform
    x: input signal, w: analysis window, N: FFT size 
    returns mX, pX: magnitude and phase spectrum
    """

    if not(UF.isPower2(N)):                                 # raise error if N not a power of two
        raise ValueError("FFT size (N) is not a power of 2")

    if (w.size > N):                                        # raise error if window size bigger than fft size
        raise ValueError("Window size (M) is bigger than FFT size")

    hN = (N//2)+1                                           # size of positive spectrum, it includes sample 0
    hM1 = (w.size+1)//2                                     # half analysis window size by rounding
    hM2 = w.size//2                                         # half analysis window size by floor
    fftbuffer = np.zeros(N)                                 # initialize buffer for FFT
    w = w / sum(w)                                          # normalize analysis window
    xw = x*w                                                # window the input sound
    fftbuffer[:hM1] = xw[hM2:]                              # zero-phase window in fftbuffer
    fftbuffer[-hM2:] = xw[:hM2]        
    X = fft(fftbuffer)                                      # compute FFT
    absX = abs(X[:hN])                                      # compute ansolute value of positive side
    absX[absX<np.finfo(float).eps] = np.finfo(float).eps    # if zeros add epsilon to handle log

    X[:hN].real[np.abs(X[:hN].real) < tol] = 0.0            # for phase calculation set to 0 the small values
    X[:hN].imag[np.abs(X[:hN].imag) < tol] = 0.0            # for phase calculation set to 0 the small values         
    pX = np.unwrap(np.angle(X[:hN]))                        # unwrapped phase spectrum of positive frequencies
    return mX, pX


def edr_slope(m,numS):
    slopes = ([])
    for t in range(numS):
        try:
            ms = m[:,t]
            rind = np.invert(np.isnan(ms))
            ms = ms[rind]
            fval = ms[0]
            mags = 10**(ms/20)
            EDR = mags**2
            ind = np.invert(np.isnan(EDR))
            rEDR = EDR[ind]
            edr = ([])
            for i in range(rEDR.shape[0]):
                edr = np.append(edr,sum(rEDR[i:]))
            log = 20*np.log10(edr)
            hsize = log.shape[0]/2
            vals = log[round(hsize-(hsize/2)):round(hsize+(hsize/2))]
            x = np.arange(0,vals.shape[0])
            model = np.polyfit(x,vals,deg=1)
            slopes = np.append(slopes,.5*model[0])
        except:
            slopes = np.append(slopes,0)
    return slopes

def Bfil(m,slopes,inceps,wt):
    beats = np.zeros(shape = m.shape)
    for i in range(m.shape[1]):
        slope = slopes[i]
        if slope==0:
            pass
        else:
            mag = m[:,i]
            ind = np.invert(np.isnan(mag))
            mag = mag[ind]
            
            x = np.arange(0,mag.shape[0])
            yint = inceps[i]
            
            edecay = x*slope+yint
            beats[:mag.shape[0],i] = (mag-edecay)*wt
            if mag.shape[0]<m.shape[0]:
                beats[mag.shape[0]:,i] = beats[mag.shape[0],i] 
    return beats

def Dfil(m,slopes,inceps,wt):    
    
    edecay = np.zeros(shape=m.shape)
    for i in range(m.shape[1]):
        slope = slopes[i]
        if slope==0:
            pass
        else:
            mag = m[:,i]
            ind = np.invert(np.isnan(mag))
            mag = mag[ind]

            x = np.arange(0,m.shape[0])
            yint = inceps[i]
            edecay[:,i] = x*(slope*wt)
    return edecay

def beat_env(wt,atk,beats):

    a = np.arange(0,wt,step=(wt/atk))

    env = np.zeros(beats.shape[0])
    env[:] = wt
    env[:a.shape[0]] = a

    for i in range(beats.shape[1]):
        beats[:,i] = env*beats[:,i]
        
    return beats

def ben_dft(x,w,H,N):
    # RUN DFT ON SOUND
    
    hM1 = int(math.floor((w.size+1)/2))                     # half analysis window size by rounding
    hM2 = int(math.floor(w.size/2))                         # half analysis window size by floor
    x = np.append(np.zeros(hM2),x)                          # add zeros at beginning to center first window at sample 0
    x = np.append(x,np.zeros(hM2))                          # add zeros at the end to analyze last sample
    pin = hM1                                               # initialize sound pointer in middle of analysis window       
    pend = x.size - hM1                                     # last sample to start a frame
    w = w / sum(w)                                          # normalize analysis window
    tfreq = np.array([])
    while pin<pend:                                         # while input sound pointer is within sound            
        x1 = x[pin-hM1:pin+hM2]                               # select frame
        mX, pX = DFT.dftAnal(x1, w, N)                        # compute dft

        if pin == hM1:                                        # if first frame initialize output sine tracks
            ffts = mX
            phs = pX
        else:                                                 # rest of frames append values to sine tracks
            ffts = np.vstack((ffts, mX))
            phs = np.vstack((phs, pX))
        pin += H
    return ffts, phs

def sinetracks(ffts, phs, w, init, x, H, t, N, fs, freqDevOffset, freqDevSlope, maxnSines, minSineDur):

    # CREATE SINE TRACKS

    mXs = ffts[init:,:]
    pXs = phs[init:,:]

    if (minSineDur <0):                          # raise error if minSineDur is smaller than 0
        raise ValueError("Minimum duration of sine tracks smaller than 0")

    hM1 = int(math.floor((w.size+1)/2))                     # half analysis window size by rounding
    hM2 = int(math.floor(w.size/2))                         # half analysis window size by floor
    x = np.append(np.zeros(hM2),x)                          # add zeros at beginning to center first window at sample 0
    x = np.append(x,np.zeros(hM2))                          # add zeros at the end to analyze last sample
    pin = hM1                                               # initialize sound pointer in middle of analysis window       
    pend = x.size - hM1                                     # last sample to start a frame
    w = w / sum(w)                                          # normalize analysis window
    tfreq = np.array([])


    for i in range(mXs.shape[0]):
        mX = mXs[i,:]
        pX = pXs[i,:]
        ploc = UF.peakDetection(mX, t)                        # detect locations of peaks
        iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)   # refine peak values by interpolation
        ipfreq = fs*iploc/float(N)                            # convert peak locations to Hertz
        # perform sinusoidal tracking by adding peaks to trajectories
        tfreq, tmag, tphase = sineTracking(ipfreq, ipmag, ipphase, tfreq, freqDevOffset, freqDevSlope)
        tfreq = np.resize(tfreq, min(maxnSines, tfreq.size))  # limit number of tracks to maxnSines
        tmag = np.resize(tmag, min(maxnSines, tmag.size))     # limit number of tracks to maxnSines
        tphase = np.resize(tphase, min(maxnSines, tphase.size)) # limit number of tracks to maxnSines
        jtfreq = np.zeros(maxnSines)                          # temporary output array
        jtmag = np.zeros(maxnSines)                           # temporary output array
        jtphase = np.zeros(maxnSines)                         # temporary output array   
        jtfreq[:tfreq.size]=tfreq                             # save track frequencies to temporary array
        jtmag[:tmag.size]=tmag                                # save track magnitudes to temporary array
        jtphase[:tphase.size]=tphase                          # save track magnitudes to temporary array
        if pin == hM1:                                        # if first frame initialize output sine tracks
            xtfreq = jtfreq 
            xtmag = jtmag
            xtphase = jtphase
        else:                                                 # rest of frames append values to sine tracks
            xtfreq = np.vstack((xtfreq, jtfreq))
            xtmag = np.vstack((xtmag, jtmag))
            xtphase = np.vstack((xtphase, jtphase))
        pin += H
    # delete sine tracks shorter than minSineDur
    xtfreq = cleaningSineTracks(xtfreq, round(fs*minSineDur/H))  
    return xtmag, xtfreq, xtphase

# CLEAN THE SINE TRACKS

def ben_clean(xtfreq,xtmag,xtphase,maxnSines):

    first = xtfreq[0,:]
    ts = np.linspace(-100,-90,num=maxnSines)
    order = np.sort(first)
    oi = np.array([],dtype=int)
    for i in range(maxnSines):
        ind = np.where(order[i]==first)[0][0]
        oi = np.append(oi,ind)

        mags = xtmag[:,i]
        try:
            thr = ts[i]
            last = min(np.where(mags<thr)[0])
            xtfreq[last:,i] = np.nan
            xtmag[last:,i] = np.nan
        except:
            pass

        zeros = np.where(xtmag[:,i]==0)[0]
        if zeros.shape[0]>0:
            size = np.where(np.isnan(xtmag[:,i])==False)[0].shape[0]
            for t in range(size):
                val = xtmag[t,i]
                if val==0:
                    xtmag[t,i] = xtmag[t-1,i] 
                    xtfreq[t,i] = xtfreq[t-1,i]

    tfreq = xtfreq[:,oi]
    tmag = xtmag[:,oi]
    tphase = xtphase[:,oi]
    
    return tfreq, tmag, tphase
