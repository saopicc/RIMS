import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import pylab
import scipy.signal
import scipy.stats
import DynSpecMS.Analysis.GeneDist
import glob
import _pickle as cPickle
from surveys_db import SurveysDB
import copy
import os
from . import ClassPlotImage
#from skimage.restoration import (denoise_wavelet, estimate_sigma)
from multiprocessing.pool import ThreadPool
import subprocess
from multiprocessing import Pool
import multiprocessing
from astropy.time import Time
from scipy import ndimage

def Save(fileout,Obj):
    cPickle.dump(Obj, open(fileout,'wb'), 2)

def imShow(I,v=5,**kwargs):
    if not "vmin" in kwargs.keys():
        RMS=scipy.stats.median_absolute_deviation(I[np.isnan(I)==False],axis=None)
        vmin=-v*RMS
        vmax= v*RMS
        kwargs["vmin"]=vmin
        kwargs["vmax"]=vmax
        
    pylab.imshow(I,interpolation="nearest",aspect="auto",**kwargs)

def Gaussian2D(x,y,GaussPar=(1.,1.,0)):
    d=np.sqrt(x**2+y**2)
    sx,sy,_=GaussPar
    return np.exp(-x**2/(2.*sx**2)-y**2/(2.*sy**2))

def doRunDir(BaseDirDB):
    BaseDir,DB=BaseDirDB
    print(BaseDir)
    CRD=ClassRunDir(BaseDir=BaseDir,DB=DB,pol=3)
    L3=CRD.runDir()

def testBruce():
    runAllDir(Patern="/home/ctasse/TestDynSpecMS/DynSpecs_1608538564",
              SaveDir="/home/ctasse/TestDynSpecMS/PNG",
              UseLoTSSDB=False)

def DBToASCII():
    with SurveysDB() as sdb:
        sdb.cur.execute('UNLOCK TABLES')
        sdb.cur.execute('select * from spectra')
        result=sdb.cur.fetchall()
    DB={}
    for t in result:
        F=t["filename"].split("/")[-1]
        DB[F]=t
        #print(t)
        
    with open('DB.txt', 'w') as f:
        for F in DB.keys():
            g=DB[F]
            f.write("%s, %s, %s \n"%(g["filename"],g["type"],g["name"]))
            if "Bright" in g["type"]:
                print(g["type"])

    
    
def runAllDir(Patern="/data/cyril.tasse/DataDynSpec_May21/*/DynSpecs_*",SaveDir=None,UseLoTSSDB=False):

    L=glob.glob(Patern)
    
    if UseLoTSSDB:
        with SurveysDB() as sdb:
            sdb.cur.execute('UNLOCK TABLES')
            sdb.cur.execute('select * from spectra')
            result=sdb.cur.fetchall()
        DB={}
        
        for t in result:
            F=t["filename"].split("/")[-1]
            DB[F]=t
    else:
        
        #DB={"1608538564_20:09:36.800_-20:26:46.000.fits":{"filename":"/data/cyril.tasse/TestDynSpecMS/DynSpecs_1608538564/TARGET/1608538564_20:09:36.800_-20:26:46.000.fits","type":"Oleg"}}
        DB={}
        LTarget=[]
        for DirName in L:
            LTarget+=glob.glob("%s/TARGET/*.fits"%DirName)
            
        for f in LTarget:
            F=f.split("/")[-1]
            DB[F]={"filename":f,"type":"NoDB"}
            
    
    
    
    
    #L=["/data/cyril.tasse/DataDynSpec_May21/P156+42/DynSpecs_L352758"]
    


    
    DBOut=[]

        
    # #tp = ThreadPool(50)
    # for iDir,BaseDir in enumerate(L):
    #     tp.apply_async(doRunDir, (BaseDir,))
    # tp.close()
    # tp.join()

    # Jobs=[(d,DB) for d in L]#[0:10]
    # with Pool(90) as p:
    #     print(p.map(doRunDir, Jobs))
    # return
    
    for iDir,BaseDir in enumerate(L):
        #if iDir<692: continue
        #if not("L658492" in BaseDir): continue
        #if not("L352758" in BaseDir): continue
        #if not("L339800" in BaseDir): continue
        #if not("L761759" in BaseDir): continue
        #if not("L259433" in BaseDir): continue
        
        print("========================== [%i / %i]"%(iDir,len(L)))
        print(BaseDir)
        CRD=ClassRunDir(BaseDir=BaseDir,DB=DB,pol=3,SaveDir=SaveDir)
        L3=CRD.runDir()
        if L3 is None:
            print("!!!!! does not have Offs")
            continue
        
        CRD=ClassRunDir(BaseDir=BaseDir,DB=DB,pol=0,SaveDir=SaveDir)
        L0=CRD.runDir()
        
        # CRD=ClassRunDir(BaseDir=BaseDir,DB=DB,pol=1)
        # L0=CRD.runDir()
        # CRD=ClassRunDir(BaseDir=BaseDir,DB=DB,pol=2)
        # L0=CRD.runDir()

        for it in range(len(L3)):
            t=L3[it]
            F=t["File"]
            tDB=copy.deepcopy(DB[F])
            
            tDB["R3"]=t["R"]
            
            # t=L0[it]
            # F=t["File"]
            # tDB["R0"]=t["R"]
            
            DBOut.append(tDB)
    Save("D.pickle",DBOut)


class ClassDist():
    def __init__(self,File,Weight="P156+42_selection/Weights.fits",pol=3,GaussPar=(3.,10.,0.),ConvMask=None):
        #print("GiveDist: Init")
        #print("open %s"%File)
        self.File=File
        F=fits.open(File)[0]
        W=fits.open(Weight)[0]
        N=W.data[0]
        npol,nch,nt=F.data.shape

        t0=F.header['OBS-STAR']
        self.StrT0=t0
        dt=F.header['CDELT1']
        dt_hours=dt/3600
        t0 = Time(t0, format='isot').mjd * 3600. * 24. + (dt/2.)
        self.times=nt
        dT_hours=dt_hours*nt
        self.fMin=float(F.header['FRQ-MIN'])
        self.fMax=float(F.header['FRQ-MAX'])
        self.extent=(0,dT_hours,self.fMin/1e6,self.fMax/1e6)

    
        sx,sy,_=GaussPar
        dxy=4.
        Nsx,Nsy=dxy*sx,dxy*sy
        xin,yin=np.mgrid[-Nsx:Nsx:(2*Nsx+1)*1j,-Nsy:Nsy:(2*Nsy+1)*1j]
        G=Gaussian2D(xin,yin,GaussPar=GaussPar)
        G/=np.sum(G)
        I=F.data[pol].copy()
        
        #I.fill(0)
        #I[nch//3,nt//2]=1
        
        N[N==0]=1
        I=I/(1./np.sqrt(N))
        #sI=I*N
        Sig=scipy.stats.median_absolute_deviation(I[I!=0],axis=None)
        
        self.Mask=(np.abs(I)>5*Sig)
        Box=11
        cMask=scipy.signal.convolve2d(self.Mask,np.ones((Box,Box),np.float32),mode="same")
        self.Mask=(np.abs(cMask)>0.1)
        self.Mask[N==0]=1
        self.Mask[I==0]=1
        
        I[self.Mask]=0.

        # NBinT=20
        # Sum0=np.sum(I,axis=0)
        # SumN0=np.sum(self.Mask,axis=0)
        # Sum0/=SumN0
        

        
        #fI = denoise_wavelet(I, channel_axis=-1, convert2ycbcr=True,
        #                     method='BayesShrink', mode='soft',
        #                     rescale_sigma=True)
        
        fI=scipy.signal.fftconvolve(I,G,mode="same")
        fI/=ConvMask
        fI[N==0]=0
        self.fI=fI
        self.I=I
        self.FracFlag=np.count_nonzero(self.Mask)/self.Mask.size

        #print("GiveDist: Init: done")
        
        
    def giveDist(self,xmm=None):
        #print("GiveDist")
        if xmm is None:
            x0,x1=self.fI.min(),self.fI.max()
        else:
            x0,x1=xmm
        DM=DynSpecMS.Analysis.GeneDist.ClassDistMachine()
        DM.setRefSample(self.fI[self.Mask==0].ravel(),Ns=100,xmm=(x0,x1))
        #print("GiveDist: Done")
        return DM.xyCumulD



    
class ClassRunDir():
    def __init__(self,BaseDir="/data/cyril.tasse/DataDynSpec_May21/P156+42/DynSpecs_L352758",
                 SaveDir="/data/cyril.tasse/VE_Py3_nancep6/TestAnalysis/PNG5",
                 DB=None,pol=3):
        self.BaseDir=BaseDir
        self.SaveDir=SaveDir
        self.DB=DB
        self.pol=pol
        self.GaussPar=(100.,15.,0.)
        #self.GaussPar=(20.,60.,0.)

        self.WeightFile=Weight="%s/Weights.fits"%BaseDir
        W=fits.open(Weight)[0]
        N=W.data[0]
        Mask=(N!=0)
        dxy=4.
        sx,sy,_=self.GaussPar
        Nsx,Nsy=dxy*sx,dxy*sy
        xin,yin=np.mgrid[-Nsx:Nsx:(2*Nsx+1)*1j,-Nsy:Nsy:(2*Nsy+1)*1j]
        G=Gaussian2D(xin,yin,GaussPar=self.GaussPar)
        self.ConvMask =scipy.signal.fftconvolve(np.float32(Mask),G,mode="same")

        

        
    def runDir(self):
        BaseDir=self.BaseDir
        GaussPar=self.GaussPar
        #BaseDir="/data/cyril.tasse/DataDynSpec_May21/P265+38/DynSpecs_L658492"

        DicoDyn=[]
        LTarget=glob.glob("%s/TARGET/*.fits"%BaseDir)#[0:1]
        LOff=glob.glob("%s/OFF/*.fits"%BaseDir)#[0:5]
        
        #BaseDir="/data/cyril.tasse/DataDynSpec_May21/P223+52/DynSpecs_L260803"
        #LTarget=glob.glob("%s/TARGET/L260803_14:52:17.890_+54:15:50.900.fitsL352758_10:23:52.117_+43:53:33.187.fits"%BaseDir)
        LTargetSel=[]

        
        
        for iF,F in enumerate(LTarget):
            ThisSpectra=self.DB.get(F.split("/")[-1],None)
            if ThisSpectra is None:
                print("!!!!!!!!!!!!!! Skipping %s"%F)
                continue
            #print(ThisSpectra["type"])
            if "Bright" in ThisSpectra["type"]:
                print("!!!!!!!!!!!!!! Skipping %s [%s]"%(F,ThisSpectra["type"]))
                continue
            LTargetSel.append(F)

        LTarget=LTargetSel
        
        Min,Max=1e10,-1e10
        for iF,F in enumerate(LTarget):
            CD=ClassDist(F,pol=self.pol,
                         GaussPar=GaussPar,
                         Weight=self.WeightFile,
                         ConvMask=self.ConvMask)
            DicoDyn.append({"CD":CD,
                            "File":F,
                            "Type":"Target"})
            x0,x1=CD.fI.min(),CD.fI.max()
            if x0<Min: Min=x0
            if x1>Max: Max=x1

        if len(LOff)<5: return None
        Min,Max=1e10,-1e10
        for iF,F in enumerate(LOff):
            CD=ClassDist(F,pol=self.pol,
                         GaussPar=GaussPar,
                         Weight=self.WeightFile,
                         ConvMask=self.ConvMask)
            DicoDyn.append({"CD":CD,
                            "File":F,
                            "Type":"Off"})
            x0,x1=CD.fI.min(),CD.fI.max()
            x0,x1=scipy.stats.mstats.mquantiles(CD.fI.ravel(),[0.1,0.9])
            if x0<Min: Min=x0
            if x1>Max: Max=x1

        
        Min=np.max([-1000,3*Min])
        Max=np.min([1000,3*Max])
        
        LyTarget=[]
        LyOff=[]
        for k in range(len(DicoDyn)):
            x,y=DicoDyn[k]["CD"].giveDist(xmm=(Min,Max))
            #print("Dist",DicoDyn[k]["File"])
            if DicoDyn[k]["Type"]=="Target":
                LyTarget.append(y)
            else:
                LyOff.append(y)
            DicoDyn[k]["Dist"]=(x,y)
            
        CumulTarget=np.array(LyTarget)
        CumulOff=np.array(LyOff)
    
        mCumulOff=np.mean(CumulOff,axis=0)
        ind=np.where((mCumulOff>0.2)&(mCumulOff<0.8))[0]
        #Std=np.sqrt(np.sum(((CumulOff-mCumulOff)[ind])**2))/ind.size
        d=(CumulOff-mCumulOff.reshape((1,-1)))
        Std=np.std(d[:,ind])
        
        #fig=pylab.figure("Dist",figsize=(7,12))
        #fig.clf()
        ListOut=[]
        
        Image="%s/../image_full_low_stokesV.dirty.fits"%self.BaseDir
        #print("Image",Image)
        if os.path.isfile(Image):
            CPI=ClassPlotImage.ClassPlotImage(Image)

        Offs=np.array([DicoDyn[i]["CD"].fI for i in range(len(DicoDyn)) if DicoDyn[i]["Type"]=="Off"])
        MeanOff=0#np.mean(Offs,axis=0)
        #Offs=np.array([DicoDyn[i]["CD"].fI for i in range(len(DicoDyn)) if DicoDyn[i]["Type"]=="Off"])
        StdOff=np.std(Offs,axis=0)
        
        for i in range(CumulTarget.shape[0]):
            R=np.sum((CumulTarget[i]-mCumulOff)**2)/x.size/Std
            FileName=DicoDyn[i]["CD"].File.split("/")[-1]
            ListOut.append({"File":FileName,
                            "R":R})
            
            #print(DicoDyn[i]["CD"].File.split("/")[-1],R)
            #if R<0.1 or DicoDyn[i]["CD"].FracFlag>0.4: continue
            
            current = 0#multiprocessing.current_process()._identity[0]
            fig = pylab.figure("DynSpecMS%i"%(current),constrained_layout=True,figsize=(8,8))
            gs = fig.add_gridspec(4, 3)
            
            fig.clf()
            #pylab.subplot(1,3,1)
            ax = fig.add_subplot(gs[0,:])
            I=DicoDyn[i]["CD"].I.copy()
            I[DicoDyn[i]["CD"].Mask]=np.nan
            imShow(I,extent=DicoDyn[i]["CD"].extent)
            pylab.ylabel("Frequency [MHz]")
            #print(DicoDyn[i]["CD"].I)
            
            ax = fig.add_subplot(gs[1,:])
            fI=DicoDyn[i]["CD"].fI.copy()
            fI[DicoDyn[i]["CD"].Mask]=np.nan
            imShow(fI,extent=DicoDyn[i]["CD"].extent)
            pylab.ylabel("Frequency [MHz]")
            pylab.xlabel("Time [hours since %s]"%(DicoDyn[i]["CD"].StrT0.replace("T"," @ ")))
            
            ax = fig.add_subplot(gs[3,:])
            fIn=(DicoDyn[i]["CD"].fI.copy()-MeanOff)/StdOff
            fIn[DicoDyn[i]["CD"].Mask]=np.nan
            imShow(fIn,extent=DicoDyn[i]["CD"].extent,vmin=-5,vmax=5)
            pylab.ylabel("Frequency [MHz]")
            pylab.xlabel("Time [hours since %s]"%(DicoDyn[i]["CD"].StrT0.replace("T"," @ ")))
            
            #imShow(DicoDyn[i]["CD"].Mask)
            
            ax = fig.add_subplot(gs[2,0])

            for j in range(CumulOff.shape[0]):
                pylab.plot(x,CumulOff[j],color="gray")
            pylab.plot(x,CumulTarget[i],color="black")
            
            if os.path.isfile(Image):
                ra,dec=self.DB[FileName]["ra"],self.DB[FileName]["decl"]
                CPI.Plot(fig,gs[2,1],ra,dec,BoxArcSec=200,pol=0)
                CPI.Plot(fig,gs[2,2],ra,dec,BoxArcSec=200,pol=1)
                
                # ax = fig.add_subplot(gs[2,2])
                # CPI=ClassPlotImage.ClassPlotImage(Image,pol=1)
                # CPI.Plot(ax)
            
            pylab.suptitle("[%s] %s, R=%f"%(self.DB[FileName]["type"],FileName,R))
            pylab.tight_layout()
            pylab.draw()
            
            pylab.show(block=False)
            pylab.pause(0.5)

            FitsName=LTarget[i].split("/")[-1]
            os.system("mkdir -p %s"%self.SaveDir)
            FName="%s/%s.pol%i.png"%(self.SaveDir,FitsName,self.pol)
            print("Saving fig: %s"%FName)
            
            fig.savefig(FName)

        return ListOut

    # Std=np.std(CumulOff,axis=0)

    # NOff,Np=CumulOff.shape
    # Chi2Off=np.zeros((NOff,),dtype=np.float32)
    # for i in range(NOff):
    #     Chi2[i]=np.sum((CumulOff[i])**2/Std**2)
        
    # pylab.figure("Im")
    # pylab.clf()
    # ax=pylab.subplot(2,1,1)
    # imShow(CD.I)
    # #pylab.imshow(I,interpolation="nearest")
    # #pylab.imshow(G,interpolation="nearest")
    
    # pylab.subplot(2,1,2,sharex=ax,sharey=ax)
    # imShow(CD.fI)
    # #pylab.subplot(3,1,3,sharex=ax,sharey=ax)
    # #imShow(N)
    # pylab.show(block=False)


    
    

        
    
        
        
