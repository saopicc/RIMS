from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import str
from builtins import range
from builtins import object
from pyrap.tables import table
import sys
from DDFacet.Other import logger
log=logger.getLogger("DynSpecMS")
from DDFacet.Array import shared_dict
from DDFacet.Other.AsyncProcessPool import APP, WorkerProcessError
from DDFacet.Other import Multiprocessing
from DDFacet.Other import ModColor
from DDFacet.Other.progressbar import ProgressBar
import numpy as np
from astropy.time import Time
from astropy import constants as const
import os
from killMS.Other import reformat
from DDFacet.Other import AsyncProcessPool
from dynspecms_version import version
import glob
from astropy.io import fits
from astropy.wcs import WCS
from DDFacet.ToolsDir.rad2hmsdms import rad2hmsdms

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

class ClassDynSpecMS(object):
    def __init__(self,
                 ListMSName=None,
                 ColName="DATA",
                 ModelName="PREDICT_KMS",
                 UVRange=[1.,1000.], 
                 ColWeights=None, 
                 SolsName=None,
                 FileCoords="Transient_LOTTS.csv",
                 Radius=3.,
                 NOff=-1,
                 ImageI=None,
                 ImageV=None,
                 SolsDir=None,
                 NCPU=1,
                 BaseDirSpecs=None,BeamModel=None,BeamNBand=1):

        self.ColName    = ColName
        self.ModelName  = ModelName
        self.ColWeights = ColWeights
        self.BeamNBand  = BeamNBand
        self.UVRange    = UVRange
        self.Mode="Spec"
        self.BaseDirSpecs=BaseDirSpecs
        self.NOff=NOff
        self.SolsName=SolsName
        self.NCPU=NCPU
        self.BeamModel=BeamModel
        
        if ListMSName is None:
            print(ModColor.Str("WORKING IN REPLOT MODE"), file=log)
            self.Mode="Plot"
            
            
        self.Radius=Radius
        self.ImageI = ImageI
        self.ImageV = ImageV
        self.SolsDir=SolsDir
        #self.PosArray=np.genfromtxt(FileCoords,dtype=[('Name','S200'),("ra",np.float64),("dec",np.float64),('Type','S200')],delimiter="\t")

        # identify version in logs
        print("DynSpecMS version %s starting up" % version(), file=log)
        self.FileCoords=FileCoords
        
        if self.Mode=="Spec":
            self.ListMSName = sorted(ListMSName)#[0:2]
            self.nMS         = len(self.ListMSName)
            self.OutName    = self.ListMSName[0].split("/")[-1].split("_")[0]
            self.ReadMSInfos()
            self.InitFromCatalog()

        elif self.Mode=="Plot":
            self.OutName    = self.BaseDirSpecs.split("_")[-1]
            self.InitFromSpecs()

    def InitFromSpecs(self):
        print("Initialising from precomputed spectra", file=log)
        ListTargetFits=glob.glob("%s/TARGET/*.fits"%self.BaseDirSpecs)#[0:1]
        F=fits.open(ListTargetFits[0])
        _,self.NChan, self.NTimes = F[0].data.shape
        t0=F[0].header['OBS-STAR']
        dt=F[0].header['CDELT1']
        t0 = Time(t0, format='isot').mjd * 3600. * 24. + (dt/2.)
        self.times=np.arange(self.NTimes)*dt+t0

        self.fMin=float(F[0].header['FRQ-MIN'])
        self.fMax=float(F[0].header['FRQ-MAX'])

        self.NDirSelected=len(ListTargetFits)
        NOrig=len(ListTargetFits)
        
        ListOffFits=glob.glob("%s/OFF/*.fits"%self.BaseDirSpecs)
        NOff=len(ListOffFits)
        
        self.PosArray=np.zeros((self.NDirSelected+NOff,),dtype=[('Name','S200'),("ra",np.float64),("dec",np.float64),('Type','S200')])
        self.PosArray=self.PosArray.view(np.recarray)
        self.PosArray.Type[len(ListTargetFits)::]="Off"
        self.NDir=self.PosArray.shape[0]
        print("For a total of %i targets"%(self.NDir), file=log)
        self.GOut=np.zeros((self.NDir,self.NChan, self.NTimes, 4), np.complex128)
        W=fits.open("%s/Weights.fits"%self.BaseDirSpecs)[0].data

        for iDir,File in enumerate(ListTargetFits+ListOffFits):
            print("  Reading %s"%File, file=log)
            F=fits.open(File)
            d=F[0].data
            ra=float(F[0].header['RA_RAD'])
            if ra<0.: ra+=2.*np.pi
            self.PosArray.ra[iDir]=ra
            dec=self.PosArray.dec[iDir]=float(F[0].header['DEC_RAD'])
            
            # print File,rad2hmsdms(ra,Type="ra").replace(" ",":"),rad2hmsdms(dec,Type="dec").replace(" ",":")
            # if self.PosArray.Type[iDir]=="Off": stop
            for iPol in range(4):
                self.GOut[iDir,:,:,iPol]=d[iPol][:,:]

        r=1./3600*np.pi/180
        if self.FileCoords:
            print("Matching ra/dec with original catalogue", file=log)
            PosArrayTarget=np.genfromtxt(self.FileCoords,dtype=[('Name','S200'),("ra",np.float64),("dec",np.float64),('Type','S200')],delimiter=",")[()]
            PosArrayTarget=PosArrayTarget.view(np.recarray)
            PosArrayTarget.ra*=np.pi/180
            PosArrayTarget.dec*=np.pi/180
            for iDir in range(self.PosArray.dec.shape[0]):
                dra=self.PosArray.ra[iDir]-PosArrayTarget.ra
                ddec=self.PosArray.dec[iDir]-PosArrayTarget.dec
                d=np.sqrt(dra**2+ddec**2)
                iS=np.argmin(d)
                if d[iS]>r:
                    print(ModColor.Str("DID NOT FIND A MATCH FOR A SOURCE"), file=log)
                    continue
                self.PosArray.Type[iDir]=PosArrayTarget.Type[iS]
                self.PosArray.Name[iDir]=PosArrayTarget.Name[iS]
                
    def InitFromCatalog(self):

        FileCoords=self.FileCoords
        dtype=[('Name','S200'),("ra",np.float64),("dec",np.float64),('Type','S200')]
        # should we use the surveys DB?
        if 'DDF_PIPELINE_DATABASE' in os.environ:
            print("Using the surveys database", file=log)
            from surveys_db import SurveysDB
            with SurveysDB() as sdb:
                sdb.cur.execute('select * from transients')
                result=sdb.cur.fetchall()
            # convert to a list, then to ndarray, then to recarray
            l=[]
            for r in result:
                l.append((r['id'],r['ra'],r['decl'],r['type']))
            if FileCoords is not None:
                print('Adding data from file '+FileCoords, file=log)
                additional=np.genfromtxt(FileCoords,dtype=dtype,delimiter=",")[()]
                if not additional.shape:
                    # deal with a one-line input file
                    additional=np.array([additional],dtype=dtype)
                for r in additional:
                    l.append(tuple(r))
            self.PosArray=np.asarray(l,dtype=dtype)
            print("Created an array with %i records" % len(result), file=log)

        else:
            
            #FileCoords="Transient_LOTTS.csv"
            if FileCoords is None:
                if not os.path.isfile(FileCoords):
                    ssExec="wget -q --user=anonymous ftp://ftp.strw.leidenuniv.nl/pub/tasse/%s -O %s"%(FileCoords,FileCoords)
                    print("Downloading %s"%FileCoords, file=log)
                    print("   Executing: %s"%ssExec, file=log)
                    os.system(ssExec)
            log.print("Reading cvs file: %s"%FileCoords)
            #self.PosArray=np.genfromtxt(FileCoords,dtype=dtype,delimiter=",")[()]
            self.PosArray=np.genfromtxt(FileCoords,dtype=dtype,delimiter=",")
            
        self.PosArray=self.PosArray.view(np.recarray)
        self.PosArray.ra*=np.pi/180.
        self.PosArray.dec*=np.pi/180.
        Radius=self.Radius
        NOrig=self.PosArray.Name.shape[0]
        Dist=AngDist(self.ra0,self.PosArray.ra,self.dec0,self.PosArray.dec)
        ind=np.where(Dist<(Radius*np.pi/180))[0]
        self.PosArray=self.PosArray[ind]
        self.NDirSelected=self.PosArray.shape[0]

        print("Selected %i target [out of the %i in the original list]"%(self.NDirSelected,NOrig), file=log)
        if self.NDirSelected==0:
            print(ModColor.Str("   Have found no sources - returning"), file=log)
            self.killWorkers()
            return
        
        NOff=self.NOff
        
        if NOff==-1:
            NOff=self.PosArray.shape[0]*2
        if NOff is not None:
            print("Including %i off targets"%(NOff), file=log)
            self.PosArray=np.concatenate([self.PosArray,self.GiveOffPosArray(NOff)])
            self.PosArray=self.PosArray.view(np.recarray)
        self.NDir=self.PosArray.shape[0]
        print("For a total of %i targets"%(self.NDir), file=log)


        self.DicoDATA = shared_dict.create("DATA")
        self.DicoGrids = shared_dict.create("Grids")
        self.DicoGrids["GridLinPol"] = np.zeros((self.NDir,self.NChan, self.NTimes, 4), np.complex128)
        self.DicoGrids["GridWeight"] = np.zeros((self.NDir,self.NChan, self.NTimes, 4), np.complex128)


        

        self.DoJonesCorr_kMS =False
        self.DicoJones=None
        if self.SolsName:
            self.DoJonesCorr_kMS=True
            self.DicoJones_kMS=shared_dict.create("DicoJones_kMS")

        self.DoJonesCorr_Beam=False
        if self.BeamModel:
            self.DoJonesCorr_Beam=True
            self.DicoJones_Beam=shared_dict.create("DicoJones_Beam")


        APP.registerJobHandlers(self)
        AsyncProcessPool.init(ncpu=self.NCPU,affinity="disable")
        APP.startWorkers()


    def GiveOffPosArray(self,NOff):
        print("Making random off catalog with %i directions"%NOff, file=log)
        CatOff=np.zeros((NOff,),self.PosArray.dtype)
        CatOff=CatOff.view(np.recarray)
        CatOff.Type="Off"
        NDone=0
        while NDone<NOff:
            dx=np.random.rand(1)[0]*self.Radius*np.pi/180
            dy=np.random.rand(1)[0]*self.Radius*np.pi/180
            ra=self.ra0+dx
            dec=self.dec0+dy
            d=AngDist(self.ra0,ra,self.dec0,dec)
            if d<self.Radius*np.pi/180:
                CatOff.ra[NDone]=ra
                CatOff.dec[NDone]=dec
                CatOff.Name[NDone]="Off%4.4i"%NDone
                NDone+=1
        return CatOff

    def ReadMSInfos(self):
        DicoMSInfos = {}

        MSName=self.ListMSName[0]
        t0  = table(MSName, ack=False)
        tf0 = table("%s::SPECTRAL_WINDOW"%self.ListMSName[0], ack=False)
        self.ChanWidth = tf0.getcol("CHAN_WIDTH").ravel()[0]
        tf0.close()
        self.times = np.unique(t0.getcol("TIME"))
        t0.close()

        tField = table("%s::FIELD"%MSName, ack=False)
        self.ra0, self.dec0 = tField.getcol("PHASE_DIR").ravel() # radians!
        if self.ra0<0.: self.ra0+=2.*np.pi
        tField.close()

        pBAR = ProgressBar(Title="Reading metadata")
        pBAR.render(0, self.nMS)
   
        #for iMS, MSName in enumerate(sorted(self.ListMSName)):
        for iMS, MSName in enumerate(self.ListMSName):
            try:
                t = table(MSName, ack=False)
            except Exception as e:
                s = str(e)
                DicoMSInfos[iMS] = {"Readable": False,
                                    "Exception": s}
                pBAR.render(iMS+1, self.nMS)
                continue

            if self.ColName not in t.colnames():
                DicoMSInfos[iMS] = {"Readable": False,
                                    "Exception": "Missing Data colname %s"%self.ColName}
                pBAR.render(iMS+1, self.nMS)
                continue

            if self.ColWeights and (self.ColWeights not in t.colnames()):
                DicoMSInfos[iMS] = {"Readable": False,
                                    "Exception": "Missing Weights colname %s"%self.ColWeights}
                pBAR.render(iMS+1, self.nMS)
                continue

            
            if  self.ModelName and (self.ModelName not in t.colnames()):
                DicoMSInfos[iMS] = {"Readable": False,
                                    "Exception": "Missing Model colname %s"%self.ModelName}
                pBAR.render(iMS+1, self.nMS)
                continue
            

            tf = table("%s::SPECTRAL_WINDOW"%MSName, ack=False)
            ThisTimes = np.unique(t.getcol("TIME"))
            if not np.allclose(ThisTimes, self.times):
                raise ValueError("should have the same times")
            DicoMSInfos[iMS] = {"MSName": MSName,
                            "ChanFreq":   tf.getcol("CHAN_FREQ").ravel(),  # Hz
                            "ChanWidth":  tf.getcol("CHAN_WIDTH").ravel(), # Hz
                            "times":      ThisTimes,
                            "startTime":  Time(ThisTimes[0]/(24*3600.), format='mjd', scale='utc').isot,
                            "stopTime":   Time(ThisTimes[-1]/(24*3600.), format='mjd', scale='utc').isot,
                            "deltaTime":  (ThisTimes[-1] - ThisTimes[0])/3600., # h
                            "Readable":   True}
            if DicoMSInfos[iMS]["ChanWidth"][0] != self.ChanWidth:
                raise ValueError("should have the same chan width")
            pBAR.render(iMS+1, self.nMS)
            
        for iMS in range(self.nMS):
            if not DicoMSInfos[iMS]["Readable"]:
                print(ModColor.Str("Problem reading %s"%MSName), file=log)
                print(ModColor.Str("   %s"%DicoMSInfos[iMS]["Exception"]), file=log)
                

        t.close()
        tf.close()
        self.DicoMSInfos = DicoMSInfos
        self.FreqsAll    = np.array([DicoMSInfos[iMS]["ChanFreq"] for iMS in list(DicoMSInfos.keys()) if DicoMSInfos[iMS]["Readable"]])
        self.Freq_minmax = np.min(self.FreqsAll), np.max(self.FreqsAll)
        self.NTimes      = self.times.size
        f0, f1           = self.Freq_minmax
        self.NChan       = int((f1 - f0)/self.ChanWidth) + 1

        # Fill properties
        self.tStart = DicoMSInfos[0]["startTime"]
        self.tStop  = DicoMSInfos[0]["stopTime"] 
        self.fMin   = self.Freq_minmax[0]
        self.fMax   = self.Freq_minmax[1]

        self.iCurrentMS=0


    
    def LoadNextMS(self):
        iMS=self.iCurrentMS
        if not self.DicoMSInfos[iMS]["Readable"]: 
            print("Skipping [%i/%i]: %s"%(iMS+1, self.nMS, self.ListMSName[iMS]), file=log)
            self.iCurrentMS+=1
            return "NotRead"
        print("Reading [%i/%i]: %s"%(iMS+1, self.nMS, self.ListMSName[iMS]), file=log)

        MSName=self.ListMSName[self.iCurrentMS]
        
        t         = table(MSName, ack=False)
        data      = t.getcol(self.ColName)
        if self.ModelName:
            print("  Substracting %s from %s"%(self.ModelName,self.ColName), file=log)
            data-=t.getcol(self.ModelName)
            
        if self.ColWeights:
            print("  Reading weight column %s"%(self.ColWeights), file=log)
            weights   = t.getcol(self.ColWeights)
        else:
            nrow,nch,_=data.shape
            weights=np.ones((nrow,nch),np.float32)
        
        flag   = t.getcol("FLAG")
        times  = t.getcol("TIME")
        A0, A1 = t.getcol("ANTENNA1"), t.getcol("ANTENNA2")
        u, v, w = t.getcol("UVW").T
        t.close()
        d = np.sqrt(u**2 + v**2 + w**2)
        uv0, uv1         = np.array(self.UVRange) * 1000
        indUV = np.where( (d<uv0)|(d>uv1) )[0]
        flag[indUV, :, :] = 1 # flag according to UV selection
        data[flag] = 0 # put down to zeros flagged visibilities

        f0, f1           = self.Freq_minmax
        nch  = self.DicoMSInfos[iMS]["ChanFreq"].size

        # Considering another position than the phase center
        u0 = u.reshape( (-1, 1, 1) )
        v0 = v.reshape( (-1, 1, 1) )
        w0 = w.reshape( (-1, 1, 1) )
        self.DicoDATA["iMS"]=self.iCurrentMS
        self.DicoDATA["data"]=data
        self.DicoDATA["weights"]=weights
        self.DicoDATA["flag"]=flag
        self.DicoDATA["times"]=times
        self.DicoDATA["A0"]=A0
        self.DicoDATA["A1"]=A1
        self.DicoDATA["u"]=u0
        self.DicoDATA["v"]=v0
        self.DicoDATA["w"]=w0
        self.DicoDATA["uniq_times"]=np.unique(self.DicoDATA["times"])

            
        if self.DoJonesCorr_kMS or self.DoJonesCorr_Beam:
            self.setJones()
        self.iCurrentMS+=1

    def setJones(self):
        from DDFacet.Data import ClassJones
        from DDFacet.Data import ClassMS

        SolsName=self.SolsName
        if "[" in SolsName:
            SolsName=SolsName.replace("[","")
            SolsName=SolsName.replace("]","")
            SolsName=SolsName.split(",")
        GD={"Beam":{"Model":self.BeamModel,
                    "LOFARBeamMode":"A",
                    "DtBeamMin":5.,
                    "NBand":self.BeamNBand,
                    "CenterNorm":1},
            "Image":{"PhaseCenterRADEC":None},
            "DDESolutions":{"DDSols":SolsName,
                            "SolsDir":self.SolsDir,
                            "GlobalNorm":None,
                            "JonesNormList":"AP"},
            "Cache":{"Dir":""}
            }
        print("Reading Jones matrices solution file:", file=log)

        ms=ClassMS.ClassMS(self.DicoMSInfos[self.iCurrentMS]["MSName"],GD=GD,DoReadData=False,)
        JonesMachine = ClassJones.ClassJones(GD, ms, CacheMode=False)
        JonesMachine.InitDDESols(self.DicoDATA)
        #JJ=JonesMachine.MergeJones(self.DicoDATA["killMS"]["Jones"],self.DicoDATA["Beam"]["Jones"])
        # import killMS.Data.ClassJonesDomains
        # DomainMachine=killMS.Data.ClassJonesDomains.ClassJonesDomains()
        # if "killMS" in self.DicoDATA.keys():
        #     self.DicoDATA["killMS"]["Jones"]["FreqDomain"]=self.DicoDATA["killMS"]["Jones"]["FreqDomains"]
        # if "Beam" in self.DicoDATA.keys():
        #     self.DicoDATA["Beam"]["Jones"]["FreqDomain"]=self.DicoDATA["Beam"]["Jones"]["FreqDomains"]
        # if "killMS" in self.DicoDATA.keys() and "Beam" in self.DicoDATA.keys():
        #     JonesSols=DomainMachine.MergeJones(self.DicoDATA["killMS"]["Jones"],self.DicoDATA["Beam"]["Jones"])
        # elif "killMS" in self.DicoDATA.keys() and not ("Beam" in self.DicoDATA.keys()):
        #     JonesSols=self.DicoDATA["killMS"]["Jones"]
        # elif not("killMS" in self.DicoDATA.keys()) and "Beam" in self.DicoDATA.keys():
        #     JonesSols=self.DicoDATA["Beam"]["Jones"]

        #self.DicoJones["G"]=np.swapaxes(self.NormJones(JonesSols["Jones"]),1,3) # Normalize Jones matrices

        if self.DoJonesCorr_kMS:
            JonesSols=self.DicoDATA["killMS"]["Jones"]
            self.DicoJones_kMS["G"]=np.swapaxes(JonesSols["Jones"],1,3) # Normalize Jones matrices
            self.DicoJones_kMS['tm']=(JonesSols["t0"]+JonesSols["t1"])/2.
            self.DicoJones_kMS['ra']=JonesMachine.ClusterCat['ra']
            self.DicoJones_kMS['dec']=JonesMachine.ClusterCat['dec']
            self.DicoJones_kMS['FreqDomains']=JonesSols['FreqDomains']
            self.DicoJones_kMS['FreqDomains_mean']=np.mean(JonesSols['FreqDomains'],axis=1)
            self.DicoJones_kMS['IDJones']=np.zeros((self.NDir,),np.int32)
            for iDir in range(self.NDir):
                ra=self.PosArray.ra[iDir]
                dec=self.PosArray.dec[iDir]
                self.DicoJones_kMS['IDJones'][iDir]=np.argmin(AngDist(ra,self.DicoJones_kMS['ra'],dec,self.DicoJones_kMS['dec']))

        if self.DoJonesCorr_Beam:
            JonesSols = JonesMachine.GiveBeam(np.unique(self.DicoDATA["times"]), quiet=True,RaDec=(self.PosArray.ra,self.PosArray.dec))
            self.DicoJones_Beam["G"]=np.swapaxes(JonesSols["Jones"],1,3) # Normalize Jones matrices
            self.DicoJones_Beam['tm']=(JonesSols["t0"]+JonesSols["t1"])/2.
            self.DicoJones_Beam['ra']=self.PosArray.ra
            self.DicoJones_Beam['dec']=self.PosArray.dec
            self.DicoJones_Beam['FreqDomains']=JonesSols['FreqDomains']
            self.DicoJones_Beam['FreqDomains_mean']=np.mean(JonesSols['FreqDomains'],axis=1)

        
        # from DDFacet.Data import ClassLOFARBeam
        # GD,D={},{}
        # D["LOFARBeamMode"]="A"
        # D["DtBeamMin"]=5
        # D["NBand"]=1
        # GD["Beam"]=D
        # BeamMachine=BeamClassLOFARBeam(self.DicoMSInfos["MSName"],GD)
        # BeamMachine.InitBeamMachine()
        # BeamTimes=BM.getBeamSampleTimes()
        # return BM.EstimateBeam(BeamTimes,
        #                        ra,dec)
        


    # def StackAll(self):
    #     while self.iCurrentMS<self.nMS:
    #         self.LoadNextMS()
    #         for iTime in range(self.NTimes):
    #             for iDir in range(self.NDir):
    #                 self.Stack_SingleTimeDir(iTime,iDir)
    #     self.Finalise()

    def StackAll(self):
        while self.iCurrentMS<self.nMS:
            if self.LoadNextMS()=="NotRead": continue
            print("Making dynamic spectra...", file=log)
            for iTime in range(self.NTimes):
                APP.runJob("Stack_SingleTime:%d"%(iTime), 
                           self.Stack_SingleTime,
                           args=(iTime,))#,serial=True)
            APP.awaitJobResults("Stack_SingleTime:*", progress="Append MS %i"%self.DicoDATA["iMS"])

            # for iTime in range(self.NTimes):
            #     self.Stack_SingleTime(iTime)
       
        self.Finalise()


    def killWorkers(self):
        print("Killing workers", file=log)
        APP.terminate()
        APP.shutdown()
        Multiprocessing.cleanupShm()



    def Finalise(self):
        self.killWorkers()

        G=self.DicoGrids["GridLinPol"]
        W=self.DicoGrids["GridWeight"].copy()
        W[W == 0] = 1
        Gn = G/W 
        self.Gn=Gn

        GOut=np.zeros_like(G)
        GOut[..., 0] =   0.5*(Gn[..., 0] + Gn[..., 3]) # I = 0.5(XX + YY)
        GOut[..., 1] =   0.5*(Gn[..., 0] - Gn[..., 3]) # Q = 0.5(XX - YY) 
        GOut[..., 2] =   0.5*(Gn[..., 1] + Gn[..., 2]) # U = 0.5(XY + YX)
        GOut[..., 3] = -0.5j*(Gn[..., 1] - Gn[..., 2]) # V = -0.5i(XY - YX)
        self.GOut = GOut

    def Stack_SingleTime(self,iTime):
        for iDir in range(self.NDir):
            self.Stack_SingleTimeDir(iTime,iDir)
        
    def Stack_SingleTimeDir(self,iTime,iDir):
        ra=self.PosArray.ra[iDir]
        dec=self.PosArray.dec[iDir]

        l, m = self.radec2lm(ra, dec)
        n  = np.sqrt(1. - l**2. - m**2.)
        self.DicoDATA.reload()
        self.DicoGrids.reload()
        indRow = np.where(self.DicoDATA["times"]==self.times[iTime])[0]
        #indRow = np.where(self.DicoDATA["times"]>0)[0]
        f   = self.DicoDATA["flag"][indRow, :, :]
        d   = self.DicoDATA["data"][indRow, :, :]
        nrow,nch,_=d.shape
        weights   = (self.DicoDATA["weights"][indRow, :]).reshape((nrow,nch,1))
        A0s = self.DicoDATA["A0"][indRow]
        A1s = self.DicoDATA["A1"][indRow]
        u0  = self.DicoDATA["u"][indRow].reshape((-1,1,1))
        v0  = self.DicoDATA["v"][indRow].reshape((-1,1,1))
        w0  = self.DicoDATA["w"][indRow].reshape((-1,1,1))
        iMS  = self.DicoDATA["iMS"]


        chfreq=self.DicoMSInfos[iMS]["ChanFreq"].reshape((1,-1,1))
        chfreq_mean=np.mean(chfreq)
        # kk  = np.exp( -2.*np.pi*1j* f/const.c.value *(u0*l + v0*m + w0*(n-1)) ) # Phasing term
        #print iTime,iDir
        kk  = np.exp(-2.*np.pi*1j* chfreq/const.c.value *(u0*l + v0*m + w0*(n-1)) ) # Phasing term

        # #ind=np.where((A0s==0)&(A1s==10))[0]
        # ind=np.where((A0s!=1000))[0]
        # import pylab
        # pylab.ion()
        # pylab.clf()
        # pylab.plot(np.angle(d[ind,2,0]))
        # pylab.plot(np.angle(kk[ind,2,0].conj()))
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)

        
        f0, _ = self.Freq_minmax
        
        DicoMSInfos      = self.DicoMSInfos

        _,nch,_=self.DicoDATA["data"].shape

        dcorr=d
        if self.DoJonesCorr_kMS:
            self.DicoJones_kMS.reload()
            tm = self.DicoJones_kMS['tm']
            # Time slot for the solution
            iTJones=np.argmin(np.abs(tm-self.times[iTime]))
            iDJones=np.argmin(AngDist(ra,self.DicoJones_kMS['ra'],dec,self.DicoJones_kMS['dec']))
            _,nchJones,_,_,_,_=self.DicoJones_kMS['G'].shape
            for iFJones in range(nchJones):
                nu0,nu1=self.DicoJones_kMS['FreqDomains'][iFJones]
                fData=self.DicoMSInfos[iMS]["ChanFreq"].ravel()
                indCh=np.where((fData>=nu0) & (fData<nu1))[0]
                #iFJones=np.argmin(np.abs(chfreq_mean-self.DicoJones_kMS['FreqDomains_mean']))
                # construct corrected visibilities
                J0 = self.DicoJones_kMS['G'][iTJones, iFJones, A0s, iDJones, 0, 0]
                J1 = self.DicoJones_kMS['G'][iTJones, iFJones, A1s, iDJones, 0, 0]
                J0 = J0.reshape((-1, 1, 1))*np.ones((1, indCh.size, 1))
                J1 = J1.reshape((-1, 1, 1))*np.ones((1, indCh.size, 1))
                #dcorr[:,indCh,:] = J0.conj() * dcorr[:,indCh,:] * J1
                dcorr[:,indCh,:] = 1./J0 * dcorr[:,indCh,:] * 1./J1.conj()
            # iFJones=np.argmin(np.abs(chfreq_mean-self.DicoJones_kMS['FreqDomains_mean']))
            # # construct corrected visibilities
            # J0 = self.DicoJones_kMS['G'][iTJones, iFJones, A0s, iDJones, 0, 0]
            # J1 = self.DicoJones_kMS['G'][iTJones, iFJones, A1s, iDJones, 0, 0]
            # J0 = J0.reshape((-1, 1, 1))*np.ones((1, nch, 1))
            # J1 = J1.reshape((-1, 1, 1))*np.ones((1, nch, 1))
            # dcorr = J0.conj() * dcorr * J1

        if self.DoJonesCorr_Beam:
            self.DicoJones_Beam.reload()
            tm = self.DicoJones_Beam['tm']
            # Time slot for the solution
            iTJones=np.argmin(np.abs(tm-self.times[iTime]))
            iDJones=np.argmin(AngDist(ra,self.DicoJones_Beam['ra'],dec,self.DicoJones_Beam['dec']))
            _,nchJones,_,_,_,_=self.DicoJones_Beam['G'].shape
            for iFJones in range(nchJones):
                nu0,nu1=self.DicoJones_Beam['FreqDomains'][iFJones]
                fData=self.DicoMSInfos[iMS]["ChanFreq"].ravel()
                indCh=np.where((fData>=nu0) & (fData<nu1))[0]
                #iFJones=np.argmin(np.abs(chfreq_mean-self.DicoJones_Beam['FreqDomains_mean']))
                # construct corrected visibilities
                J0 = self.DicoJones_Beam['G'][iTJones, iFJones, A0s, iDJones, 0, 0]
                J1 = self.DicoJones_Beam['G'][iTJones, iFJones, A1s, iDJones, 0, 0]
                J0 = J0.reshape((-1, 1, 1))*np.ones((1, indCh.size, 1))
                J1 = J1.reshape((-1, 1, 1))*np.ones((1, indCh.size, 1))
                #dcorr[:,indCh,:] = J0.conj() * dcorr[:,indCh,:] * J1
                dcorr[:,indCh,:] = 1./J0 * dcorr[:,indCh,:] * 1./J1.conj()
                

            
        #ds=np.sum(d*kk, axis=0) # without Jones
        ds = np.sum(dcorr * kk*weights, axis=0) # with Jones
        ws = np.sum((1-f)*weights, axis=0)

        ich0 = int( (self.DicoMSInfos[iMS]["ChanFreq"][0] - f0)/self.ChanWidth )
        self.DicoGrids["GridLinPol"][iDir,ich0:ich0+nch, iTime, :] = ds
        self.DicoGrids["GridWeight"][iDir,ich0:ich0+nch, iTime, :] = ws


    def NormJones(self, G):
        print("  Normalising Jones matrices by the amplitude", file=log)
        G[G != 0.] /= np.abs(G[G != 0.])
        return G
        


    def radec2lm(self, ra, dec):
        # ra and dec must be in radians
        l = np.cos(dec) * np.sin(ra - self.ra0)
        m = np.sin(dec) * np.cos(self.dec0) - np.cos(dec) * np.sin(self.dec0) * np.cos(ra - self.ra0)
        return l, m
# =========================================================================
# =========================================================================
