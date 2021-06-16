import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import pylab
import scipy.signal
import scipy.stats
import DynSpecMS.Analysis.GeneDist
import glob
import _pickle as cPickle
#from surveys_db import SurveysDB
import copy
import os

def Save(fileout,Obj):
    cPickle.dump(Obj, open(fileout,'wb'), 2)

def imShow(I,v=10):
    RMS=scipy.stats.median_absolute_deviation(I,axis=None)
    pylab.imshow(I,interpolation="nearest",vmin=-v*RMS,vmax=v*RMS,aspect="auto")

def Gaussian2D(x,y,GaussPar=(1.,1.,0)):
    d=np.sqrt(x**2+y**2)
    sx,sy,_=GaussPar
    return np.exp(-x**2/(2.*sx**2)-y**2/(2.*sy**2))

class ClassDist():
    def __init__(self,File,Weight="P156+42_selection/Weights.fits",pol=3,GaussPar=(3.,10.,0.),ConvMask=None):
        #print("open %s"%File)
        self.File=File
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
        fI/=ConvMask
        fI[N==0]=0
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


def runAllDir():
    
    with SurveysDB() as sdb:
        sdb.cur.execute('UNLOCK TABLES')
        sdb.cur.execute('select * from spectra')
        result=sdb.cur.fetchall()
    DB={}
    for t in result:
        F=t["filename"].split("/")[-1]
        DB[F]=t

    
    L=glob.glob("/data/cyril.tasse/DataDynSpec_May21/*/DynSpecs_*")
    #L=["/data/cyril.tasse/DataDynSpec_May21/P156+42/DynSpecs_L352758"]
    
    # L=["/data/cyril.tasse/TestDynSpecMS/DynSpecs_1608538564"]
    # DB={"1608538564_20:09:36.800_-20:26:46.000.fits":{"filename":"/data/cyril.tasse/TestDynSpecMS/DynSpecs_1608538564/TARGET/1608538564_20:09:36.800_-20:26:46.000.fits","type":"Oleg"}}


    
    DBOut=[]

    for iDir,BaseDir in enumerate(L):
        #if iDir<692: continue
        print("========================== [%i / %i]"%(iDir,len(L)))
        print(BaseDir)
        CRD=ClassRunDir(BaseDir=BaseDir,DB=DB,pol=3)
        L3=CRD.runDir()
        if L3 is None:
            print("!!!!! does not have Offs")
            continue
        
        CRD=ClassRunDir(BaseDir=BaseDir,DB=DB,pol=0)
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
            
            t=L0[it]
            F=t["File"]
            tDB["R0"]=t["R"]
            
            DBOut.append(tDB)
    Save("D.pickle",DBOut)

    
class ClassRunDir():
    def __init__(self,BaseDir="/data/cyril.tasse/DataDynSpec_May21/P156+42/DynSpecs_L352758",
                 DB=None,pol=3):
        self.BaseDir=BaseDir
        self.SaveDir="/data/cyril.tasse/VE_Py3_nancep6/TestAnalysis/PNG"
        self.DB=DB
        self.pol=pol
        self.GaussPar=(10.,5.,0.)

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
        # BaseDir="/data/cyril.tasse/DataDynSpec_May21/P265+38/DynSpecs_L658492"

        DicoDyn=[]
        LTarget=glob.glob("%s/TARGET/*.fits"%BaseDir)
        #LTarget=glob.glob("%s/TARGET/L352758_10:23:52.117_+43:53:33.187.fits"%BaseDir)
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

        LOff=glob.glob("%s/OFF/*.fits"%BaseDir)#[0:5]
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
            
        CumulTarget=np.array(LyTarget)
        CumulOff=np.array(LyOff)
    
        mCumulOff=np.mean(CumulOff,axis=0)
        ind=np.where((mCumulOff>0.2)&(mCumulOff<0.8))[0]
        #Std=np.sqrt(np.sum(((CumulOff-mCumulOff)[ind])**2))/ind.size
        d=(CumulOff-mCumulOff.reshape((1,-1)))
        Std=np.std(d[:,ind])
        
        fig=pylab.figure("Dist")
        fig.clf()
        ListOut=[]
        
        for i in range(CumulTarget.shape[0]):
            R=np.sum((CumulTarget[i]-mCumulOff)**2)/x.size/Std
            ListOut.append({"File":DicoDyn[i]["CD"].File.split("/")[-1],
                            "R":R})
            
            #print(DicoDyn[i]["CD"].File.split("/")[-1],R)
            #if R<0.01: continue
            pylab.clf()
            pylab.subplot(1,3,1)
            imShow(DicoDyn[i]["CD"].I)
            pylab.subplot(1,3,2)
            imShow(DicoDyn[i]["CD"].fI)
            pylab.subplot(1,3,3)
            for j in range(CumulOff.shape[0]):
                pylab.plot(x,CumulOff[j],color="gray")
            pylab.plot(x,CumulTarget[i],color="black")
            pylab.suptitle("%s: %f"%(LTarget[i].split("/")[-1],R))
            pylab.draw()
            #pylab.show(block=False)
            #pylab.pause(0.5)
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


    
    

        
    
        
        
