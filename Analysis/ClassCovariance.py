import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import pylab
import DDFacet.ToolsDir.Gaussian as Gaussian
import scipy.signal
import scipy.stats
import DynSpecMS.Analysis.GeneDist

def imShow(I,v=10):
    RMS=scipy.stats.median_absolute_deviation(I,axis=None)
    pylab.imshow(I,interpolation="nearest",vmin=-v*RMS,vmax=v*RMS)

def Gaussian2D(x,y,GaussPar=(1.,1.,0)):
    d=np.sqrt(x**2+y**2)
    sx,sy,_=GaussPar
    return np.exp(-x**2/(2.*sx**2)-y**2/(2.*sy**2))

class ClassDist():
    def __init__(self,File,Weight="P156+42_selection/Weights.fits",pol=3,GaussPar=(3.,10.,0.)):
        F=fits.open(File)[0]
        W=fits.open(Weight)[0]
        N=W.data[0]
        npol,nch,nt=F.data.shape
    
        sx,sy,_=GaussPar
        dxy=4.
        Nsx,Nsy=dxy*sx,dxy*sy
        xin,yin=np.mgrid[-Nsx:Nsx:(2*Nsx+1)*1j,-Nsy:Nsy:(2*Nsy+1)*1j]
        G=Gaussian2D(xin,yin,GaussPar=GaussPar)

        I=F.data[pol].copy()
        #I.fill(0)
        #I[nch//3,nt//2]=1
        N[N==0]=1
        I=I/(1./np.sqrt(N))
        #sI=I*N
        #Sig=scipy.stats.median_absolute_deviation(sI,axis=None)
        fI=scipy.signal.fftconvolve(I,G,mode="same")
        self.fI=fI
        self.I=I
        
    def giveDist(self,xmm=None):
        if xmm is None:
            x0,x1=self.fI.min(),self.fI.max()
        else:
            x0,x1=xmm
        DM=DynSpecMS.Analysis.GeneDist.ClassDistMachine()
        DM.setRefSample(self.fI.ravel(),Ns=100,xmm=(x0,x1))
        return DM.xyCumulD
        


def test():
    GaussPar=(3.,10.,0.)
    CD=ClassDist("P156+42_selection/L352758_10:23:52.117_+43:53:33.187.fits",GaussPar=GaussPar)
    x0,x1=CD.fI.min(),CD.fI.max()
    x,y=CD.giveDist()
    
    pylab.figure("Im")
    pylab.clf()
    ax=pylab.subplot(2,1,1)
    imShow(CD.I)
    #pylab.imshow(I,interpolation="nearest")
    #pylab.imshow(G,interpolation="nearest")
    
    pylab.subplot(2,1,2,sharex=ax,sharey=ax)
    imShow(CD.fI)
    #pylab.subplot(3,1,3,sharex=ax,sharey=ax)
    #imShow(N)
    pylab.show(block=False)

    CD1=ClassDist("P156+42_selection/OFF.fits",GaussPar=GaussPar)
    x1,y1=CD1.giveDist()

    
    pylab.figure("Dist")
    pylab.clf()
    pylab.plot(x,y)
    pylab.plot(x1,y1)
    pylab.show(block=False)
    
        
        
