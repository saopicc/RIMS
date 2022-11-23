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
import matplotlib
import time
import DDFacet.Other.MyPickle
import DDFacet.Other.ClassTimeIt

from DDFacet.Other import logger
log=logger.getLogger("ClassCovariance")
from DDFacet.Array import shared_dict
from DDFacet.Other.AsyncProcessPool import APP, WorkerProcessError
from DDFacet.Other import Multiprocessing
from DDFacet.Other import AsyncProcessPool
import gc

def AngDist(ra0,ra1,dec0,dec1):
    AC=np.arccos
    C=np.cos
    S=np.sin
    D=S(dec0)*S(dec1)+C(dec0)*C(dec1)*C(ra0-ra1)

    if type(D).__name__=="ndarray":
        D[D>1.]=1.
        D[D<-1.]=-1.
    else:
        if D>1.: D=1.
        if D<-1.: D=-1.
    return AC(D)


def Save(fileout,Obj):
    cPickle.dump(Obj, open(fileout,'wb'), 2)
    
def imShow(I,v=5,MeanCorr=False,**kwargs):
    if MeanCorr:
        I=I-np.median(I[(I!=0)&(np.isnan(I)==False)])
        
    if not "vmin" in kwargs.keys():
        RMS=scipy.stats.median_absolute_deviation(I[(I!=0)&(np.isnan(I)==False)],axis=None)
        vmin=-v*RMS
        vmax= v*RMS
        kwargs["vmin"]=vmin
        kwargs["vmax"]=vmax
        
    r=pylab.imshow(I[::-1,:],interpolation="nearest",aspect="auto",**kwargs)
    
    return r

    
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
    # runAllDir(Patern="/home/ctasse/TestDynSpecMS/DynSpecs_1608538564",
    #           SaveDir="/home/ctasse/TestDynSpecMS/PNG",
    #           UseLoTSSDB=False)
    
    # runAllDir(Patern="/home/ctasse/TestDynSpecMS/DynSpecs_1622491578",
    #           SaveDir="/home/ctasse/TestDynSpecMS/PNG_DEEP2",
    #           UseLoTSSDB=False)
    
    # runAllDir(Patern="/home/ctasse/TestDynSpecMS/TestWeights_Natural",
    #           SaveDir="/home/ctasse/TestDynSpecMS/PNG_DEEP_WEIGHTS_ILV",
    #           UseLoTSSDB=False)

    # runAllDir(Patern="/home/ctasse/TestDynSpecMS/TestWeights_Natural",
    #           SaveDir="/home/ctasse/TestDynSpecMS/PNG_DEEP_Jy",
    #           UseLoTSSDB=False)
    
    # runAllDir(Patern="/home/ctasse/TestDynSpecMS/TestWeights",
    #           SaveDir="/home/ctasse/TestDynSpecMS/PNG_DEEP_Jy_briggs",
    #           UseLoTSSDB=False)

    runAllDir(Patern="/home/ctasse/TestDynSpecMS/TestWeights_WhereChan",
              SaveDir="/home/ctasse/TestDynSpecMS/PNG_DEEP_whereChan",
              UseLoTSSDB=False)
    
def DB2np(D):
    k=list(D.keys())[0]
    d=D[k]
    dtype=[(k,type(d[k])) for k in d.keys()]
    dtype=[]
    for k in d.keys():
        if isinstance(d[k],str):
            s="|S200"
        else:
            s=np.float32
        dtype.append((k,s))
    Cat=np.zeros((len(D),),dtype=dtype)
    for i,K in enumerate(D.keys()):
        #print("%i/%i"%(i,len(D)))
        for k in d.keys():
            #print(k)
            Cat[i][k]=D[K][k]
    Cat=Cat.view(np.recarray)
    return Cat
    
def PlotTargets():
    # with SurveysDB() as sdb:
    #     sdb.cur.execute('UNLOCK TABLES')
    #     sdb.cur.execute('select * from spectra')
    #     result=sdb.cur.fetchall()
    # DB={}
    # for t in result:
    #     F=t["filename"].split("/")[-1]
    #     DB[F]=t
    # DDFacet.Other.MyPickle.Save(DB,"DB.pickle")
    
    # D=DDFacet.Other.MyPickle.Load("DB.pickle")
    # Cat=DB2np(D)
    # np.save("DB.npy",Cat)

    C=np.load("DB.npy").view(np.recarray)
    import pylab
    pylab.clf()
    LType=[]
    LN=[]
    for k in np.unique(C.type):
        ka=k.decode("ascii")
        if "b'"==ka[0:2]:
            ka=ka[2:-1]
        if "Off" in ka: continue
        ind=np.where(C.type==k)
        N=ind[0].size
        # print(k,ka,N)
        if ka in LType:
            LN[LType.index(ka)]+=N
        else:
            LType.append(ka)
            LN.append(N)
        c=C[ind]
    #     pylab.plot(c.ra,c.decl,ls="",marker="o",label=ka,linewidth=0, markersize=4,markeredgewidth=0,alpha=0.5)
    # pylab.xlabel("RA (deg)")
    # pylab.ylabel("DEC (deg)")
    # pylab.legend()
    # pylab.draw()
    # pylab.show()

    from matplotlib import pyplot as plt
    # plt.pie(Popularity)
    ind=np.argsort(LN)
    LType=["%s [%i]"%(LType[i],LN[i]) for i in ind]
    LN=[LN[i] for i in ind]
    plt.pie( LN, labels = LType, autopct='%0.1f%%', explode=tuple([0.1]*len(LN)))
    plt.axis('equal')
    plt.show()

    
    
def DBToASCII():
    # with SurveysDB() as sdb:
    #     sdb.cur.execute('select * from transients')
    #     result=sdb.cur.fetchall()
    #     # convert to a list, then to ndarray, then to recarray
    # l=[]
    # for r in result:
    #     l.append((r['id'],r['ra'],r['decl'],r['type']))
    # stop

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

def runOneLOFAR():
    CRA=ClassRunAll(Patern="/data/cyril.tasse/DataDynSpec_May21/*/DynSpecs_L352758",
                   SaveDir="PNG_PRES_ONE",
                   UseLoTSSDB=True)
    CRA.run()

def runAllLOFAR():
    
    # CRA=ClassRunAll(Patern="/data/cyril.tasse/DataDynSpec_May21/P236+53/DynSpecs_L470106",
    #                 SaveDir="PNG_PRES_P236+53",
    #                 UseLoTSSDB=True)


    
    
    CRA=ClassRunAll(Patern="/data/cyril.tasse/DataDynSpec_May21/*/DynSpecs_*",
                    SaveDir="PNG_PRES_NEW",
                    UseLoTSSDB=True)
    
    CRA.run()

import regions
def appendFacetIDToDB(DB):
    #DB=DDFacet.Other.MyPickle.Load("DB_In.pickle")
    Cat=DB2np(DB)

    
    # DPointing={}
    # ll=glob.glob("/data/cyril.tasse/DataDynSpec_May21/*/*.tessel.reg")
    # T=DDFacet.Other.ClassTimeIt.ClassTimeIt()
    # for ireg,reg in enumerate(ll):
    #     print("%i/%i"%(ireg,len(ll)))
    #     R=regions.Regions.read(reg)
    #     PName=reg.split("/")[-2]
    #     Lra=[]
    #     Ldec=[]
    #     LCluster=[]
    #     for r in R:
    #         if "PointSkyRegion" in r.__str__():
    #             Lra.append(r.center.ra.deg)
    #             Ldec.append(r.center.dec.deg)
    #             LCluster.append(int(r.meta["label"].split("_")[1][:-1].replace("S","")))
    #     Cp=np.zeros((len(Lra),),dtype=[("ra",np.float32),("dec",np.float32),("Cluster",int)])
    #     Cp=Cp.view(np.recarray)
    #     Cp.ra[:]=np.array(Lra)
    #     Cp.dec[:]=np.array(Ldec)
    #     Cp.Cluster[:]=np.array(LCluster)
    #     DPointing[PName]=Cp
    #     if ireg%10==0:
    #         Save("ClusterDB.pickle",DPointing)
    #     T.timeit()
    # Save("ClusterDB.pickle",DPointing)

    DPointing=DDFacet.Other.MyPickle.Load("ClusterDB.pickle")
    for iField,field in enumerate(sorted(list(DPointing.keys()))):
        print("%i/%i"%(iField,len(DPointing)))
        ind=np.where(Cat["field"]==field.encode("UTF-8"))[0]
        Cs=Cat[ind]
        if ind.size==0:
            continue

        rap=DPointing[field]["ra"]*np.pi/180
        if DPointing[field]["ra"].size==0: continue
        decp=DPointing[field]["dec"]*np.pi/180
        
        for iSpectra,Spectra in enumerate(Cs["filename"]):
            ra=Cs["ra"][iSpectra]*np.pi/180
            dec=Cs["decl"][iSpectra]*np.pi/180
            D=AngDist(ra,rap,dec,decp)
            iFacet=np.argmin(D)
            iTessel=DPointing[field]["Cluster"][iFacet]
            key=Spectra.decode("ascii").split("/")[-1]
            DB[key]["Cluster"]=iTessel
        
        
class ClassRunAll():
    def __init__(self,
                 Patern="/data/cyril.tasse/DataDynSpec_May21/*/DynSpecs_*",
                 SaveDir=None,
                 UseLoTSSDB=False):
        self.Patern=Patern
        self.SaveDir=SaveDir
        self.UseLoTSSDB=UseLoTSSDB
        if UseLoTSSDB:
            # with SurveysDB() as sdb:
            #     sdb.cur.execute('UNLOCK TABLES')
            #     sdb.cur.execute('select * from spectra')
            #     result=sdb.cur.fetchall()
            # DB={}
            
            # for t in result:
            #     F=t["filename"].split("/")[-1]
            #     DB[F]=t
            # appendFacetIDToDB(DB)
            # for Obj in list(DB.keys()):
            #     DB[Obj]["R_I"]=0.
            #     DB[Obj]["R_V"]=0.
            #     DB[Obj]["R_L"]=0.
            #     DB[Obj]["W0_I"]=0.
            #     DB[Obj]["W0_V"]=0.
            #     DB[Obj]["W0_L"]=0.
            #     DB[Obj]["W1_I"]=0.
            #     DB[Obj]["W1_V"]=0.
            #     DB[Obj]["W1_L"]=0.
            #     DB[Obj]["MaxSig_I"]=0.
            #     DB[Obj]["MaxSig_V"]=0.
            # Save("DB_In.pickle",DB)
            # stop
            
            DB=DDFacet.Other.MyPickle.Load("DB_In.pickle")        
        else:
            #DB={"1608538564_20:09:36.800_-20:26:46.000.fits":{"filename":"/data/cyril.tasse/TestDynSpecMS/DynSpecs_1608538564/TARGET/1608538564_20:09:36.800_-20:26:46.000.fits","type":"Oleg"}}
            DB={}
            LTarget=[]
            for DirName in L:
                LTarget+=glob.glob("%s/TARGET/*.fits"%DirName)
                
            for f in LTarget:
                F=f.split("/")[-1]
                DB[F]={"filename":f,"type":"NoDB"}

        print("ok")
        self.Taper=(20,20)
        self.Taper=(40,40)
        self.Taper=(20,5)
        
                

            
        # L=["/data/cyril.tasse/DataDynSpec_May21/P156+42/DynSpecs_L352758"]
    
        
        APP.registerJobHandlers(self)
        AsyncProcessPool.init(ncpu=40,
                              affinity="disable")
        APP.startWorkers()
        
        self.DB=shared_dict.dict_to_shm("DB", DB)
        

    def run(self):
        Patern=self.Patern
        SaveDir=self.SaveDir
        UseLoTSSDB=self.UseLoTSSDB
        
        L=glob.glob(Patern)
    

        
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
            #print("========================== [%i / %i]"%(iDir,len(L)))
            #self._runDir(iDir,BaseDir)
            APP.runJob("runDir:%d"%(iDir), 
                       self._runDir,
                       args=(iDir,BaseDir))#,serial=True)

        APP.awaitJobResults("runDir:*", progress="Compute stat")

        #Save("DBOut.pickle",self.DB)
        DB=shared_dict.attach("DB")
        Cat=DB2np(DB)
        np.save("DBOut.%i_%i.npy"%(self.Taper[0],self.Taper[1]),Cat)

    def _runDir(self,iDir,BaseDir):
        try:
            self._runDir2(iDir,BaseDir)
        except Exception as e:
            print("[%s]"%self.BaseDir,e)
            print("[%s]"%self.BaseDir,e)
            print("[%s]"%self.BaseDir,e)
            print("[%s]"%self.BaseDir,e)

        
        
    def _runDir2(self,iDir,BaseDir):
        #if iDir<692: continue
        #if not("L658492" in BaseDir): continue
        #if not("L352758" in BaseDir): continue
        #if not("L339800" in BaseDir): continue
        #if not("L761759" in BaseDir): continue
        #if not("L259433" in BaseDir): continue
        #L=self.L
        DB=shared_dict.attach("DB")
        if iDir%100==0:
            Cat=DB2np(DB)
            np.save("DBOut.%i_%i.npy"%(self.Taper[0],self.Taper[1]),Cat)

        Patern=self.Patern
        SaveDir=self.SaveDir
        UseLoTSSDB=self.UseLoTSSDB
        
        #print(BaseDir)
        # CRD=ClassRunDir(BaseDir=BaseDir,DB=DB,pol="V",SaveDir=SaveDir)
        # L3=CRD.runDir()
        # if L3 is None:
        #     print("!!!!! does not have Offs")
        #     continue
        
        #CRD=ClassRunDir(BaseDir=BaseDir,DB=DB,pol="L",SaveDir=SaveDir)
        #L0=CRD.runDir()
        
        # CRD0=ClassRunDir(BaseDir=BaseDir,DB=DB,pol="Q",SaveDir=SaveDir)
        # CRD0.runDir()
        # # CRD.Plot()
        
        # CRD1=ClassRunDir(BaseDir=BaseDir,DB=DB,pol="U",SaveDir=SaveDir)
        # CRD1.runDir()
        
        CRD=ClassRunDir(BaseDir=BaseDir,DB=DB,pol="I",SaveDir=SaveDir,
                        Taper=self.Taper)
        L0=CRD.runDir()
        
        #CRD.Plot()
        
        CRD=ClassRunDir(BaseDir=BaseDir,DB=DB,pol="V",SaveDir=SaveDir)
        L0=CRD.runDir()
        
        # # CRD=ClassRunDir(BaseDir=BaseDir,DB=DB,pol=1)
        # # L0=CRD.runDir()
        # # CRD=ClassRunDir(BaseDir=BaseDir,DB=DB,pol=2)
        # # L0=CRD.runDir()
        
        # for it in range(len(L3)):
        #     t=L3[it]
        #     F=t["File"]
        #     tDB=copy.deepcopy(DB[F])
        
        #     tDB["R3"]=t["R"]
        
        #     # t=L0[it]
        #     # F=t["File"]
        #     # tDB["R0"]=t["R"]
        
        #     DBOut.append(tDB)
            


class ClassDist():
    def __init__(self,File,Weight="P156+42_selection/Weights.fits",pol="I",GaussPar=(3.,10.,0.),
                 ConvMask=None,
                 InMask=None,
                 ApplyFilter=True):
        #print("GiveDist: Init")
        #print("open %s"%File)
        self.File=File
        fitsF=fits.open(File)
        fitsW=fits.open(Weight)
        F=fitsF[0]
        W=fitsW[0]
        Fdata=F.data
        Wdata=W.data
        N=Wdata[0]
        #self.Name=F.header['NAME']
        npol,nch,nt=Fdata.shape

        t0=F.header['OBS-STAR']
        self.StrT0=t0
        dt=F.header['CDELT1']
        dt_hours=dt/3600
        t0 = Time(t0, format='isot').mjd * 3600. * 24. + (dt/2.)
        self.times=nt


        self.nt=nt
        dT_hours=dt_hours*nt
        
        self.fMin=float(F.header['FRQ-MIN'])
        self.fMax=float(F.header['FRQ-MAX'])
        self.nch=nch
        self.extent=(0,dT_hours,self.fMin/1e6,self.fMax/1e6)
        
        sx,sy,_=GaussPar
        dxy=4.
        Nsx,Nsy=dxy*sx,dxy*sy
        xin,yin=np.mgrid[-Nsx:Nsx:(2*Nsx+1)*1j,-Nsy:Nsy:(2*Nsy+1)*1j]
        G=Gaussian2D(xin,yin,GaussPar=GaussPar)
        G/=np.sum(G)

        if pol=="I":
            I0=Fdata[0].copy()
        elif pol=="Q":
            I0=Fdata[1].copy()
        elif pol=="U":
            I0=Fdata[2].copy()
        elif pol=="V":
            I0=Fdata[3].copy()
        elif pol=="L":
            I0=np.sqrt(Fdata[1].copy()**2+Fdata[2].copy()**2)
        I0*=1e3
        # I.fill(0)
        # I[nch//3,nt//2]=1
        
        #N[N==0]=1
        w=np.sqrt(N)
        # print("!!!!")
        # w.fill(1)
        I=I0*w

        #sI=I*N
        self.I=I

        if InMask is None:
            self.computeMask()
        else:
            self.Mask=InMask
        
        self.Mask[N==0]=1
        self.Mask[I==0]=1
        
        I[self.Mask]=0.
        w[self.Mask]=0.
        N[self.Mask]=0
        #if np.all(self.Mask): stop
        
        I=I0*w
        
        
        # NBinT=20
        # Sum0=np.sum(I,axis=0)
        # SumN0=np.sum(self.Mask,axis=0)
        # Sum0/=SumN0
        

        
        #fI = denoise_wavelet(I, channel_axis=-1, convert2ycbcr=True,
        #                     method='BayesShrink', mode='soft',
        #                     rescale_sigma=True)
        
        if ApplyFilter:
            fI=scipy.signal.fftconvolve(I,G,mode="same")
            ws=scipy.signal.fftconvolve(w,G,mode="same")
            fI[ws==0]=0
            ws[ws==0]=1
            fI/=ws

            #fI/=ConvMask
        
            fI[N==0]=0
            self.fI=fI
            M=(fI==0)
            m=np.mean(fI[M==0])
            
            #self.fI-=m
            #self.fI[M]=0
        self.I=I
        self.N=N
        self.FracFlag=np.count_nonzero(self.Mask)/self.Mask.size
        fitsF.close()
        fitsW.close()
        del(Fdata,Wdata)
        gc.collect()
        #print("GiveDist: Init: done")

    def computeMask(self):
        I=self.I
        Sig=scipy.stats.median_absolute_deviation(I[I!=0],axis=None)
        self.Mask=(np.abs(I)>20*Sig)
        BoxTime=3
        BoxFreq=1
        cMask=scipy.signal.convolve2d(self.Mask,np.ones((BoxFreq,BoxTime),np.float32),mode="same")
        self.Mask=(np.abs(cMask)>0.1)
        # print("!!!!")
        # # self.Mask.fill(0)
        # self.Mask=(I==0)

        
        
    def giveDist(self,xmm=None):
        #print("GiveDist")
        if xmm is None:
            x0,x1=self.fI.min(),self.fI.max()
        else:
            x0,x1=xmm
        #return None,None
        DM=DynSpecMS.Analysis.GeneDist.ClassDistMachine()
        DM.setRefSample(self.fI[self.Mask==0].ravel(),Ns=100,xmm=(x0,x1))
        
        #print("GiveDist: Done")
        #stop
        return DM.xyCumulD



    

    
class ClassRunDir():
    def __init__(self,BaseDir="/data/cyril.tasse/DataDynSpec_May21/P156+42/DynSpecs_L352758",
                 SaveDir="/data/cyril.tasse/VE_Py3_nancep6/TestAnalysis/PNG5",
                 DB=None,pol="I",InMask=None,Taper=(20,20)):
        #print("Doing pol %s"%pol)
        self.InMask=InMask
        self.BaseDir=BaseDir
        self.SaveDir=SaveDir
        self.DB=DB
        self.pol=pol
        self.GaussPar=(Taper[0],Taper[1],0.)
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
        # BaseDir="/data/cyril.tasse/DataDynSpec_May21/P265+38/DynSpecs_L658492"

        DicoDyn=[]
        LTarget0=glob.glob("%s/TARGET/*.fits"%BaseDir)#[0:1]
        LTarget=[]
        for File in LTarget0:
            F=fits.open(File)[0]

            try:
                Name=F.header['NAME']
            except:
                Name=""
            if "Control" in Name:
                print("Skipping %s"%Name)
                continue
            
            LTarget.append(File)
        if len(LTarget)==0:
            print("%s/TARGET: THERE ARE NO TARGET!!!!!"%self.BaseDir)
            print("%s/TARGET: THERE ARE NO TARGET!!!!!"%self.BaseDir)
            print("%s/TARGET: THERE ARE NO TARGET!!!!!"%self.BaseDir)
            print("%s/TARGET: THERE ARE NO TARGET!!!!!"%self.BaseDir)
            return
        
        LOff=glob.glob("%s/OFF/*.fits"%BaseDir)#[0::2]
        import datetime
        if len(LOff)==0:
            print("%s/OFF: THERE ARE NO OFF!!!!!"%self.BaseDir)
            print("%s/OFF: THERE ARE NO OFF!!!!!"%self.BaseDir)
            print("%s/OFF: THERE ARE NO OFF!!!!!"%self.BaseDir)
            print("%s/OFF: THERE ARE NO OFF!!!!!"%self.BaseDir)
            return
        datetime_object = datetime.datetime.now()
        #print(datetime_object)
        
        #print("There %i target"%len(LTarget))
        #print("There %i off"%len(LOff))
        
        # BaseDir="/data/cyril.tasse/DataDynSpec_May21/P223+52/DynSpecs_L260803"
        # LTarget=glob.glob("%s/TARGET/L260803_14:52:17.890_+54:15:50.900.fitsL352758_10:23:52.117_+43:53:33.187.fits"%BaseDir)
        LTargetSel=[]

        L_CD=[]
        L_M=[]
        for iF,F in enumerate(LOff):
            CD=ClassDist(F,pol=self.pol,
                         GaussPar=GaussPar,
                         Weight=self.WeightFile,
                         ConvMask=self.ConvMask,
                         ApplyFilter=False)
            L_M.append(CD.Mask)
        if self.InMask is None:
            Mask=np.any(np.array(L_M),axis=0)
            self.Mask=Mask
        else:
            Mask=self.Mask=self.InMask
            
        
        for iF,F in enumerate(LTarget):
            ThisSpectra=self.DB.get(F.split("/")[-1],None)
            if ThisSpectra is None:
                #print("!!!!!!!!!!!!!! Skipping %s"%F)
                continue
            #print(ThisSpectra["type"])
            if "Bright" in ThisSpectra["type"]:
                #print("!!!!!!!!!!!!!! Skipping %s [%s]"%(F,ThisSpectra["type"]))
                continue
            LTargetSel.append(F)

        LTarget=LTargetSel
        if len(LTarget)==0:
            print("%s/TARGET: No target after DB filtering!!!!!"%self.BaseDir)
            print("%s/TARGET: No target after DB filtering!!!!!"%self.BaseDir)
            print("%s/TARGET: No target after DB filtering!!!!!"%self.BaseDir)
            print("%s/TARGET: No target after DB filtering!!!!!"%self.BaseDir)
            return
        
        Min,Max=1e10,-1e10
        for iF,F in enumerate(LTarget):
            CD=ClassDist(F,pol=self.pol,
                         GaussPar=GaussPar,
                         Weight=self.WeightFile,
                         InMask=Mask,
                         ConvMask=self.ConvMask)
            DicoDyn.append({"CD":CD,
                            "File":F,
                            "Type":"Target"})
            x0,x1=CD.fI.min(),CD.fI.max()
            if x0<Min: Min=x0
            if x1>Max: Max=x1

        #if len(LOff)<5: return None
        #Min,Max=1e10,-1e10
        for iF,F in enumerate(LOff):
            try:
                CD=ClassDist(F,pol=self.pol,
                             GaussPar=GaussPar,
                             Weight=self.WeightFile,
                             InMask=Mask,
                             ConvMask=self.ConvMask)
            except Exception as e:
                print("[%]"%self.BaseDir,self.e)
                print("[%]"%self.BaseDir,self.e)
                print("[%]"%self.BaseDir,self.e)
                print("[%]"%self.BaseDir,self.e)
                continue
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
        self.NOff=0
        self.NTarget=0
        for k in range(len(DicoDyn)):
            x,y=DicoDyn[k]["CD"].giveDist(xmm=(Min,Max))
            #print("Dist",DicoDyn[k]["File"])
            if DicoDyn[k]["Type"]=="Target":
                LyTarget.append(y)
                self.NTarget+=1
            else:
                LyOff.append(y)
                self.NOff+=1
            DicoDyn[k]["Dist"]=(x,y)
        dX=x[1]-x[0]
        self.CumulTarget=np.array(LyTarget)
        self.CumulOff=np.array(LyOff)
        self.mCumulOff=np.mean(self.CumulOff,axis=0)
        self.DicoDyn=DicoDyn

        ind=np.where((self.mCumulOff>0.2)&(self.mCumulOff<0.8))[0]
        
        
        d=(self.CumulOff-self.mCumulOff.reshape((1,-1)))

        StdMaxX=np.std(self.CumulOff,axis=0)
        StdMaxX[StdMaxX==0]=np.min(StdMaxX[StdMaxX!=0])
        
        Std=np.std(d[:,ind])
        Np=self.CumulTarget.shape[1]
        try:
            Np=self.CumulTarget.shape[1]
        except:
            print("self.CumulTarget",self.CumulTarget,self.BaseDir)
            print("self.CumulTarget",self.CumulTarget,self.BaseDir)
            print("self.CumulTarget",self.CumulTarget,self.BaseDir)
            return

        
        for i in range(len(DicoDyn)):
            F=DicoDyn[i]["File"].split("/")[-1]
            x,ThisCumul=DicoDyn[i]["Dist"]
            R=np.sum((ThisCumul-self.mCumulOff)**2)/Np/Std
            # W=np.sum(np.abs(ThisCumul.reshape((1,-1))-self.CumulOff),axis=1)
            # W=np.sqrt( np.sum( ((ThisCumul.reshape((1,-1))-self.CumulOff)/StdMaxX.reshape((1,-1)))**2,axis=1))
            W=( np.sum( np.abs((ThisCumul.reshape((1,-1))-self.CumulOff)/StdMaxX.reshape((1,-1)))*dX,axis=1))

            
            W=W[W!=0]
            if F in self.DB.keys():
                self.DB[F]["R_%s"%self.pol]=R
                SigMAD=scipy.stats.median_abs_deviation(W,scale="normal")
                self.DB[F]["W0_%s"%self.pol]=np.mean(W)/SigMAD
                self.DB[F]["W1_%s"%self.pol]=np.median(W)/SigMAD
                
            #print(self.DB[F])
            
        Offs=np.array([DicoDyn[i]["CD"].fI for i in range(len(DicoDyn)) if DicoDyn[i]["Type"]=="Off"])
        MeanOff=0
        StdOff=np.std(Offs,axis=0)
        StdOff[StdOff==0]=1
        for i in range(len(DicoDyn)):
            F=DicoDyn[i]["File"].split("/")[-1]
            fIn=(DicoDyn[i]["CD"].fI.copy()-MeanOff)/StdOff
            fIn[DicoDyn[i]["CD"].Mask]=0
            Max=np.max(np.abs(fIn))
            if F in self.DB.keys():
                self.DB[F]["MaxSig_%s"%self.pol]=Max
        


            

    def GiveTimeSliceSmooth(self,iTarget,TimeFreq,Type="Time"):
        t0,t1,f0,f1=self.DicoDyn[iTarget]["CD"].extent
        nch=self.DicoDyn[iTarget]["CD"].nch
        nt=self.DicoDyn[iTarget]["CD"].nt
        times=np.linspace(t0,t1,nt)
        freqs=np.linspace(f0,f1,nch)
        Offs=np.array([self.DicoDyn[i]["CD"].fI for i in range(len(self.DicoDyn)) if self.DicoDyn[i]["Type"]=="Off"])
        sOff=np.std(Offs,axis=0)
        
        # M=np.zeros_like(Offs)
        # M[Offs!=0]=1
        # s0=np.sum(np.sum(Offs**2,axis=0),axis=1)
        # s1=np.sum(np.sum(M,axis=0),axis=1)
        # s1[s0==0]=1
        # sOff=np.sqrt(s0/s1)
        
        if Type=="Time":
            it=np.argmin(np.abs(TimeFreq-times))
            x=freqs
            y=self.DicoDyn[iTarget]["CD"].fI[:,it]
            
            ey=sOff[:,it]
            #ey=sOff[:]
            
        return x,y,ey
        
    def GiveTimeSlice(self,iTarget,t,Dt,nFreq):
        t0,t1,f0,f1=self.DicoDyn[iTarget]["CD"].extent
        nch=self.DicoDyn[iTarget]["CD"].nch
        nt=self.DicoDyn[iTarget]["CD"].nt
        times=np.linspace(t0,t1,nt)
        freqs=np.linspace(f0,f1,nch)

        dt=times[1]-times[0]
        nDt=np.ceil((Dt/dt)//2)
        it=np.argmin(np.abs(t-times))
        it0=int(np.max([0,it-nDt]))
        it1=int(np.min([nt,it+nDt]))
        iFreq=np.int16(np.linspace(0,nch,nFreq+1))
        
        x=freqs

        Offs=np.array([self.DicoDyn[i]["CD"].I for i in range(len(self.DicoDyn)) if self.DicoDyn[i]["Type"]=="Off"])
        Lx=[]
        Ly=[]
        Lex=[]
        Ley=[]
        for iBin in range(nFreq):
            if0,if1=iFreq[iBin],iFreq[iBin+1]
            fm=(freqs[if0]+freqs[if1-1])/2
            Lx.append(fm)
            Lex.append((freqs[if0]-freqs[if1-1])/2.)
            y=self.DicoDyn[iTarget]["CD"].I[if0:if1,it0:it1]
            n=self.DicoDyn[iTarget]["CD"].N[if0:if1,it0:it1]
            w=np.sqrt(n)
            y*=w
            Ly.append(np.sum(y)/np.sum(w))
            ey=Offs[:,if0:if1,it0:it1]
            we=w.reshape((1,w.shape[0],w.shape[1]))
            
            ey*=we
            
            sey=np.sum(np.sum(ey,axis=-1),axis=-1)
            swe=np.sum(np.sum(we,axis=-1),axis=-1)
            sey=sey/swe
            Ley.append(np.std(sey))
            

            
        return np.array(Lx),np.array(Lex),np.array(Ly),np.array(Ley)
        
    def Plot(self):
        CumulTarget=self.CumulTarget
        CumulOff=self.CumulOff
        DicoDyn=self.DicoDyn
        
        mCumulOff=np.mean(CumulOff,axis=0)
        ind=np.where((mCumulOff>0.2)&(mCumulOff<0.8))[0]
        #Std=np.sqrt(np.sum(((CumulOff-mCumulOff)[ind])**2))/ind.size
        d=(CumulOff-mCumulOff.reshape((1,-1)))
        Std=np.std(d[:,ind])
        
        # fig=pylab.figure("Dist",figsize=(7,12))
        # fig.clf()
        
        ListOut=[]
        
        Image="%s/../image_full_low_stokesV.dirty.fits"%self.BaseDir
        #print("Image",Image)
        if os.path.isfile(Image):
            CPI=ClassPlotImage.ClassPlotImage(Image)

        Offs=np.array([DicoDyn[i]["CD"].fI for i in range(len(DicoDyn)) if DicoDyn[i]["Type"]=="Off"])
        MeanOff=0#np.mean(Offs,axis=0)
        #Offs=np.array([DicoDyn[i]["CD"].fI for i in range(len(DicoDyn)) if DicoDyn[i]["Type"]=="Off"])
        StdOff=np.std(Offs,axis=0)
        StdOff[StdOff==0]=1
        for i in range(len(DicoDyn)):
            if DicoDyn[i]["Type"]=="Off": continue
            R=np.sum((CumulTarget[i]-mCumulOff)**2)/self.CumulTarget.shape[1]/Std
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
            MeanCorr=False
            if self.pol=="L": MeanCorr=True

            cmap="seismic"
            imShow(I,MeanCorr=MeanCorr,extent=DicoDyn[i]["CD"].extent)#,cmap="gray")

            pylab.ylabel("Frequency [MHz]")
            #print(DicoDyn[i]["CD"].I)
            
            ax = fig.add_subplot(gs[1,:],sharex=ax,sharey=ax)
            fI=DicoDyn[i]["CD"].fI.copy()
            fI[DicoDyn[i]["CD"].Mask]=np.nan
            imShow(fI,MeanCorr=MeanCorr,extent=DicoDyn[i]["CD"].extent,cmap=cmap)
            pylab.ylabel("Frequency [MHz]")
            pylab.xlabel("Time [hours since %s]"%(DicoDyn[i]["CD"].StrT0.replace("T"," @ ")))
            
            ax = fig.add_subplot(gs[3,:],sharex=ax,sharey=ax)
            fIn=(DicoDyn[i]["CD"].fI.copy()-MeanOff)/StdOff
            fIn[DicoDyn[i]["CD"].Mask]=0
            Max=np.max(np.abs(fIn))
            fIn[DicoDyn[i]["CD"].Mask]=np.nan
            imShow(fIn,MeanCorr=MeanCorr,extent=DicoDyn[i]["CD"].extent,vmin=-5,vmax=5,cmap=cmap)
            pylab.ylabel("Frequency [MHz]")
            pylab.xlabel("Time [hours since %s]"%(DicoDyn[i]["CD"].StrT0.replace("T"," @ ")))
            
            #imShow(DicoDyn[i]["CD"].Mask)
            
            ax = fig.add_subplot(gs[2,0])

            for k in range(len(DicoDyn)):
                if DicoDyn[k]["Type"]=="Target": continue
                x,y=DicoDyn[k]["Dist"]
                pylab.plot(x,y,color="gray")
            pylab.plot(x,CumulTarget[i],color="black")
            
            if os.path.isfile(Image):
                ra,dec=self.DB[FileName]["ra"],self.DB[FileName]["decl"]
                CPI.Plot(fig,gs[2,1],ra,dec,BoxArcSec=200,pol=0)
                CPI.Plot(fig,gs[2,2],ra,dec,BoxArcSec=200,pol=1)
                
                # ax = fig.add_subplot(gs[2,2])
                # CPI=ClassPlotImage.ClassPlotImage(Image,pol=1)
                # CPI.Plot(ax)
            
            pylab.suptitle("[%s] %s, R=%f, Max=%f"%(self.DB[FileName]["type"],FileName,R,Max))
            pylab.tight_layout()
            pylab.draw()
            
            pylab.show(block=False)
            pylab.pause(0.5)
            FitsName=FileName#LTarget[i].split("/")[-1]
            os.system("mkdir -p %s"%self.SaveDir)
            FName="%s/%s.%s.png"%(self.SaveDir,FitsName,self.pol)
            print("Saving fig: %s"%FName)
            
            fig.savefig(FName)

        return ListOut

