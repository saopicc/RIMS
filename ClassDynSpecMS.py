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
from DDFacet.Other import AsyncProcessPool

from DDFacet.Other import Multiprocessing
from DDFacet.Other import ModColor
from DDFacet.Other.progressbar import ProgressBar
import numpy as np
from astropy.time import Time
from DDFacet.Other import ClassTimeIt
from astropy import constants as const
import os
from killMS.Other import reformat
from DDFacet.Other import AsyncProcessPool
#from .dynspecms_version import version
from dynspecms_version import version
import glob
from astropy.io import fits
from astropy.wcs import WCS
from DDFacet.ToolsDir.rad2hmsdms import rad2hmsdms
from DDFacet.ToolsDir import ModCoord
from SkyModel.Array import RecArrayOps
import DDFacet.Other.MyPickle
import Polygon
from Polygon.Utils import convexHull
#import DynSpecMS.testLibBeam
#from killMS.Data import ClassJonesDomains
import DDFacet.Other.ClassJonesDomains
import psutil

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

def StrToList(s):
    if isinstance(s,list): return s
    if isinstance(s,str):
        s=s.strip()
        if s=="None": return None
        Ls=s.split(",")
        Ls[0]=Ls[0].replace("[","")
        Ls[-1]=Ls[-1].replace("]","")
        for iss,ss in enumerate(Ls):
            Ls[iss]=float(ss)
    return Ls

class ClassDynSpecMS(object):
    def __init__(self,
                 ListMSName=None,
                 ColName="DATA",
                 TChunkHours=0.,
                 ModelName="PREDICT_KMS",
                 UVRange=[1.,1000.],
                 ColWeights=None,
                 SolsName=None,
                 FileCoords="Transient_LOTTS.csv",
                 Radius=3.,
                 NOff=-1,
                 DicoFacet=None,
                 ImageI=None,
                 ImageV=None,
                 SolsDir=None,
                 NCPU=1,
                 BaseDirSpecs=None,BeamModel=None,BeamNBand=1,
                 SourceCatOff=None,
                 SourceCatOff_FluxMean=None,
                 SourceCatOff_dFluxMean=None,
                 options=None):
        self.options=options
        # CutGainsMinMax

        if BeamModel=="None":
            BeamModel=None
        if SolsName=="None":
            SolsName=None
        self.DFacet=None
        self.SourceCatOff_FluxMean=SourceCatOff_FluxMean
        self.SourceCatOff_dFluxMean=SourceCatOff_dFluxMean
        self.SourceCatOff=SourceCatOff
        self.DicoFacet=DicoFacet
        self.ColName    = ColName
        if ModelName=="None": ModelName=None
        self.ModelName  = ModelName
        self.TChunkHours=TChunkHours
        self.ColWeights = ColWeights
        self.BeamNBand  = BeamNBand
        self.UVRange    = UVRange
        self.Mode="Spec"
        self.BaseDirSpecs=BaseDirSpecs
        self.NOff=NOff
        self.SolsName=SolsName
        self.NCPU=NCPU
        self.BeamModel=BeamModel
        self.ra0, self.dec0=None,None
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

        self.PosArray=np.zeros((self.NDirSelected+NOff,),dtype=[('Name','S200'),("ra",np.float64),
                                                                ("dec",np.float64),('Type','S200'),
                                                                ('iTessel',int),('iFacet',int)])
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
        DoProperMotionCorr=False
        if self.options.UseLoTSSDB:
            print("Using the surveys database", file=log)
            from surveys_db import SurveysDB
            with SurveysDB() as sdb:
                sdb.cur.execute('select * from transients')
                result=sdb.cur.fetchall()
            # convert to a list, then to ndarray, then to recarray
            l=[]
            for r in result:
                l.append((r['id'],r['ra'],r['decl'],r['type']))

            if FileCoords is not None and FileCoords!="":
                print('Adding data from file '+FileCoords, file=log)
                additional=np.genfromtxt(FileCoords,dtype=dtype,delimiter=",")[()]
                if len(additional.shape)==0: additional=additional.reshape((1,))
                if not additional.shape:
                    # deal with a one-line input file
                    additional=np.array([additional],dtype=dtype)
                for r in additional:
                    l.append(tuple(r))

            self.PosArray=np.asarray(l,dtype=dtype)
            print("Created an array with %i records" % len(result), file=log)
        elif self.options.UseGaiaDB is not None:
            from astroquery.gaia import Gaia
            rac_deg,decc_deg=self.ra0*180/np.pi, self.dec0*180/np.pi
            Radius_deg=self.Radius
            Dmax,NMax=self.options.UseGaiaDB.split(",")
            Dmax,NMax=float(Dmax),int(NMax)
            Parallax_min=1./(Dmax*1e-3)
            query=f"""SELECT TOP 10000 gaia_source.designation,gaia_source.source_id,gaia_source.ref_epoch,gaia_source.ra,gaia_source.dec,gaia_source.parallax,
            gaia_source.pmra,gaia_source.pmdec,gaia_source.phot_g_mean_mag,gaia_source.bp_rp
            FROM gaiadr3.gaia_source
            WHERE
            CONTAINS(
	    POINT('ICRS',gaiadr3.gaia_source.ra,gaiadr3.gaia_source.dec),
	    CIRCLE('ICRS',{rac_deg},{decc_deg},{Radius_deg})
            )=1  AND  (gaiadr3.gaia_source.parallax>={Parallax_min})"""
            log.print(f"Sending du Gaia server")
            print(f"{query}")
            job = Gaia.launch_job(query)
            result = job.get_results()

            log.print(f"Query has returned {len(result)}")



            l=[]
            dtype=[('Name','S200'),("ra",np.float64),("dec",np.float64),
                   ("pmra",np.float64),("pmdec",np.float64),("ref_epoch",np.float64),
                   ("parallax",np.float64),("GaiaDistance",np.float64),
                   ("g",np.float64),("G",np.float64),("b_r",np.float64),
                   ('Type','S200')]
            for r in result:
                parallax=r['parallax']
                g=r['g']
                D=1./(parallax*1e-3)
                G=g+5*np.log10(parallax)-10
                l.append((r['DESIGNATION'],r['ra'],r['dec'],r['pmra'],r['pmdec'],r['ref_epoch'],
                          r['parallax'],D,
                          g,G,r['b_r'],
                          b"Gaia DR3"))
            self.PosArray=np.asarray(l,dtype=dtype)


            CGGS=ClassGiveGaiaSample((rac_deg,decc_deg,Radius_deg),self.PosArray,RefCat="/data/cyril.tasse/Analyse_DataDynSpec_Jan23_TestRM/MergedCat.npz.MergeGaia.npz")
            indExo,indGaia=CGGS.buildRandGaiaSample()
            stop
            # ##################################
            F=fits.open(self.options.FitsCatalog)
            d=F[1].data
            l=[]
            dtype=[('Name','S200'),("ra",np.float64),("dec",np.float64),
                   ("pmra",np.float64),("pmdec",np.float64),("ref_epoch",np.float64),("parallax",np.float64),
                   ('Type','S200')]
            for r in d:
                l.append((r['DESIGNATION'],r['ra'],r['dec'],r['pmra'],r['pmdec'],r['ref_epoch'],r['parallax'],r['Type']))

            if FileCoords is not None and FileCoords!="":
                print('Adding data from file '+FileCoords, file=log)
                dtype0=[('Name','S200'),("ra",np.float64),("dec",np.float64),('Type','S200')]
                additional=np.genfromtxt(FileCoords,dtype=dtype0,delimiter=",")[()]
                if len(additional.shape)==0: additional=additional.reshape((1,))
                if not additional.shape:
                    # deal with a one-line input file
                    additional=np.array([additional],dtype=dtype0)
                additional1=np.zeros((additional.shape[0],),dtype=dtype)
                additional1["Name"][:]=additional["Name"][:]
                additional1["ra"][:]=additional["ra"][:]
                additional1["dec"][:]=additional["dec"][:]
                additional1["Type"][:]=additional["Type"][:]

                for r in additional1:
                    l.append(tuple(r))


            self.PosArray=np.asarray(l,dtype=dtype)
            log.print("Created an array with %i records" % len(l))
            DoProperMotionCorr=True


            # import pylab
            # pylab.clf()
            # pylab.subplot(1,2,1)
            # pylab.scatter(self.PosArray["ra"],self.PosArray["dec"])
            # pylab.scatter([rac_deg],[decc_deg])
            # pylab.subplot(1,2,2)
            # D=1./(self.PosArray["parallax"]*1e-3)
            # pylab.hist(D,bins=100)
            # pylab.draw()
            # pylab.show()

            DoProperMotionCorr=True
            if self.PosArray.size>NMax:
                ind=np.int64(np.random.rand(NMax)*self.PosArray.size)
                self.PosArray=self.PosArray[ind]

            log.print("Created an array with %i records" % self.PosArray.size)

        elif self.options.FitsCatalog:
            print("Using the fits catalog: %s"%self.options.FitsCatalog, file=log)
            F=fits.open(self.options.FitsCatalog)
            d=F[1].data
            l=[]
            dtype=[('Name','S200'),("ra",np.float64),("dec",np.float64),
                   ("pmra",np.float64),("pmdec",np.float64),("ref_epoch",np.float64),("parallax",np.float64),
                   ('Type','S200')]
            for r in d:
                l.append((r['DESIGNATION'],r['ra'],r['dec'],r['pmra'],r['pmdec'],r['ref_epoch'],r['parallax'],r['Type']))

            if FileCoords is not None and FileCoords!="":
                print('Adding data from file '+FileCoords, file=log)
                dtype0=[('Name','S200'),("ra",np.float64),("dec",np.float64),('Type','S200')]
                additional=np.genfromtxt(FileCoords,dtype=dtype0,delimiter=",")[()]
                if len(additional.shape)==0: additional=additional.reshape((1,))
                if not additional.shape:
                    # deal with a one-line input file
                    additional=np.array([additional],dtype=dtype0)
                additional1=np.zeros((additional.shape[0],),dtype=dtype)
                additional1["Name"][:]=additional["Name"][:]
                additional1["ra"][:]=additional["ra"][:]
                additional1["dec"][:]=additional["dec"][:]
                additional1["Type"][:]=additional["Type"][:]

                for r in additional1:
                    l.append(tuple(r))


            self.PosArray=np.asarray(l,dtype=dtype)
            log.print("Created an array with %i records" % len(l))
            DoProperMotionCorr=True
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
            if len(self.PosArray.shape)==0: self.PosArray=self.PosArray.reshape((1,))

        self.PosArray=self.PosArray.view(np.recarray)
        self.PosArray.ra*=np.pi/180.
        self.PosArray.dec*=np.pi/180.

        Radius=self.Radius
        NOrig=self.PosArray.Name.shape[0]
        Dist=AngDist(self.ra0,self.PosArray.ra,self.dec0,self.PosArray.dec)
        ind=np.where(Dist<(Radius*np.pi/180))[0]
        self.PosArray=self.PosArray[ind]

        if DoProperMotionCorr:
            Lra,Ldec=[],[]
            log.print("Do proper motion corrrection")
            for iPCat,PCat in enumerate(self.PosArray):
                #print(iPCat,len(self.PosArray))
                ra,dec=PCat["ra"],PCat["dec"]

                ra1,dec1=ProperMotionCorrection(PCat["ra"],PCat["dec"],
                                                PCat["pmra"],PCat["pmdec"],PCat["ref_epoch"],PCat["parallax"],self.tmin)
                if np.isnan(ra1) or np.isnan(dec1):
                    ra1,dec1=ra,dec
                    log.print(str((PCat["ra"],PCat["dec"],PCat["pmra"],PCat["pmdec"],PCat["ref_epoch"],PCat["parallax"],ra1,dec1)))

                Lra.append(ra1-ra)
                Ldec.append(dec1-dec)
                self.PosArray["ra"][iPCat]=ra1
                self.PosArray["dec"][iPCat]=dec1

            print(np.array(Lra)*3600.*180/np.pi,np.array(Ldec)*3600.*180/np.pi)


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


        if self.DFacet is not None:
            Rrad0=(0.001)*np.pi/180#self.Radius*np.pi/180
            theta=np.linspace(0,2*np.pi,10)
            xC0=np.cos(theta)*Rrad0
            yC0=np.sin(theta)*Rrad0

            def give_iFacet_iTessel(l,m):
                #Plm=Polygon.Polygon(np.array([[l,m]]))
                Plm=Polygon.Polygon(np.array([xC0+l,yC0+m]).T)
                iFacetMax=-1
                ArrMax=-1
                for iFacet in self.DFacet.keys():
                    P=Polygon.Polygon(self.DFacet[iFacet]["Polygon"])
                    Arr=(P&Plm).area()
                    if Arr>ArrMax: #P.covers(Plm):
                        ArrMax=Arr
                        iFacetMax=iFacet
                return iFacetMax,self.DFacet[iFacetMax]["iSol"]

            self.PosArray=RecArrayOps.AppendField(self.PosArray,'iFacet',int)
            self.PosArray=RecArrayOps.AppendField(self.PosArray,'iTessel',int)
            l, m = self.CoordMachine.radec2lm(self.PosArray.ra, self.PosArray.dec)

            for iThis in range(self.PosArray.size):
                iFacet,iSol=give_iFacet_iTessel(l[iThis],m[iThis])
                self.PosArray.iFacet[iThis]=iFacet
                self.PosArray.iTessel[iThis]=iSol

            # import pylab
            # for iTessel in np.unique(self.PosArray.iTessel):
            #     ind=np.where(self.PosArray.iTessel==iTessel)[0]
            #     rap,decp=self.CoordMachine.lm2radec(self.PosArray.ra, self.PosArray.dec)
            #     pylab.scatter(self.PosArray.ra[ind],self.PosArray.dec[ind])#,c=iTessel)
            # pylab.draw()
            # pylab.show()

        self.NDir=self.PosArray.shape[0]
        print("For a total of %i targets"%(self.NDir), file=log)

        #self.radecToReg(self.PosArray.ra,self.PosArray.dec,self.PosArray.Type)


        #self.DicoDATA = shared_dict.create("DATA")
        self.DicoGrids = shared_dict.create("Grids")

        self.DicoGrids["GridLinPol"] = np.zeros((self.NDir,self.NChan, self.NTimesGrid, 4), np.complex128)
        self.DicoGrids["GridWeight"] = np.zeros((self.NDir,self.NChan, self.NTimesGrid, 4), np.complex128)
        self.DicoGrids["GridWeight2"] = np.zeros((self.NDir,self.NChan, self.NTimesGrid, 4), np.complex128)




        self.DoJonesCorr_kMS =False
        self.DicoJones=None
        if self.SolsName:
            self.DoJonesCorr_kMS=True

        self.DoJonesCorr_Beam=False
        if self.BeamModel:
            self.DoJonesCorr_Beam=True

        AsyncProcessPool.APP=None
        # AsyncProcessPool.init(ncpu=self.NCPU,
        #                       num_io_processes=1,
        #                       affinity="disable")
        AsyncProcessPool.init((self.NCPU or psutil.cpu_count(logical=False)-2),
                              affinity=0,
                              num_io_processes=1,
                              verbose=0)
        self.APP=AsyncProcessPool.APP

        self.APP.registerJobHandlers(self)
        self.APP.startWorkers()





    def GiveOffPosArray(self,NOff):
        print("Making random off catalog with %i directions"%NOff, file=log)
        CatOff=np.zeros((NOff,),self.PosArray.dtype)
        CatOff=CatOff.view(np.recarray)
        CatOff.Type="Off"
        NDone=0
        if self.SourceCatOff is not None and self.SourceCatOff!="":
            log.print(ModColor.Str("Reading off sources catalog: %s"%(self.SourceCatOff), color="green"))
            F=fits.open(self.SourceCatOff)
            Fd= F[1].data
            Fd=Fd.view(np.recarray)
            Fd=Fd[Fd.S_Code==b"S"]
            SMean=self.SourceCatOff_FluxMean
            dSMean=self.SourceCatOff_dFluxMean
            S0,S1=SMean-dSMean,SMean+dSMean
            ind=np.where( (Fd.Isl_Total_flux>S0) & (Fd.Isl_Total_flux<S1) )[0]
            # ind=np.where( Fd.Isl_Total_flux == Fd.Isl_Total_flux.max())[0]
            Fd=Fd[ind]

            log.print("There are %i selected off sources with flux in [%f, %f] Jy"%(ind.size,S0,S1))
            if Fd.RA[0]<0:
                Fd.RA+=360.

            for iS in range(NOff):

                iSel=int(np.random.rand(1)[0]*ind.size)
                CatOff.ra[iS]=Fd.RA[iSel]*np.pi/180
                CatOff.dec[iS]=Fd.DEC[iSel]*np.pi/180
                CatOff.Name[iS]="Off%4.4i"%iS
        elif self.DicoFacet is not None and self.DicoFacet!="":
            self.DFacet=DFacet=DDFacet.Other.MyPickle.Load(self.DicoFacet)
            DicoDir={}
            # for iFacet in list(DFacet.keys()):
            #     iSol=DFacet[iFacet]["iSol"][0]
            #     if not iSol in list(DicoDir.keys()):
            #         DicoDir[iSol]=[iFacet]
            #     else:
            #         DicoDir[iSol].append(iFacet)

            for iFacet in list(DFacet.keys()):
                iSol=iFacet
                DicoDir[iSol]=[iFacet]
            DicoPolyTessel={}

            Rrad=self.Radius*np.pi/180
            theta=np.linspace(0,2*np.pi,1000)
            xC=np.cos(theta)*Rrad
            yC=np.sin(theta)*Rrad
            Pc=Polygon.Polygon(np.array([xC,yC]).T)


            for iTessel in DicoDir.keys():
                iFacet=DicoDir[iTessel][0]
                P=Polygon.Polygon(DFacet[iFacet]["Polygon"])
                for iFacet in DicoDir[iTessel][1:]:
                    P+=Polygon.Polygon(DFacet[iFacet]["Polygon"])
                Ph=convexHull(P)
                Pint=(Pc&Ph)
                if Pint.area()>0:
                    DicoPolyTessel[iTessel]=Ph

            def ClosePolygon(polygon):
                P = np.array(polygon).tolist()
                polygon = np.array(P + [P[0]])
                return Polygon.Polygon(polygon)

            def give_in_points(P,NRand=5):
                P=ClosePolygon(P[0])
                #P=Polygon.Polygon(np.array(P[0])*1000)

                x,y=np.array(P).T
                x0,x1=x.min(),x.max()
                y0,y1=y.min(),y.max()
                Lx=[]
                Ly=[]
                # import pylab
                # pylab.clf()
                # xxx,yyy=np.array(P).T
                # pylab.plot(xxx,yyy)

                Rrad0=10/3600*np.pi/180 # the off source will be chosen to be further than 10" from the edge of the domain
                ArrRrad0=np.pi*Rrad0**2
                theta=np.linspace(0,2*np.pi,10)
                xC0=np.cos(theta)*Rrad0
                yC0=np.sin(theta)*Rrad0



                for iDone in range(NRand):
                    while True:
                        xx=float(np.random.rand(1)[0]*(x1-x0)+x0)
                        yy=float(np.random.rand(1)[0]*(y1-y0)+y0)
                        #print(P)
                        #P1=Polygon.Polygon(np.array([[xx,yy]],np.float64))
                        Pc0=ClosePolygon(Polygon.Polygon(np.array([xC0+xx,yC0+yy]).T)[0])
                        Arr=(P&Pc0).area()
                        if Arr/(Pc0.area())>0.95:#.covers(P1):
                            # xxc,yyc=np.array(Pc0).T
                            # pylab.plot(xxc,yyc,color="blue")
                            # pylab.draw()
                            # pylab.show(block=False)
                            # pylab.pause(0.5)
                            # print("kkkkkk")
                            break

                        #else:
                            # pylab.scatter(xx,yy,color="red")
                            # pylab.draw()
                            # pylab.show(block=False)
                            # pylab.pause(0.5)
                            # print("kkk")
                    Lx.append(xx)
                    Ly.append(yy)

                return np.array(Lx),np.array(Ly)



            NPerTessel=np.max([self.options.nMinOffPerFacet,self.NOff//len(DicoPolyTessel)])
            log.print("Using %i off sources per facet"%NPerTessel)
            NDone=0

            NOff=NPerTessel*len(DicoPolyTessel)
            CatOff=np.zeros((1000,),self.PosArray.dtype)
            CatOff=CatOff.view(np.recarray)
            CatOff.Type="Off"

            for iNode in DicoPolyTessel.keys():
                P=Polygon.Polygon(DicoPolyTessel[iNode])
                #print(iNode,P)
                l,m=give_in_points(P,NRand=NPerTessel)

                ra, dec = self.CoordMachine.lm2radec(np.array(l), np.array(m))
                for iThis in range(ra.size):
                    CatOff.ra[NDone]=ra[iThis]
                    CatOff.dec[NDone]=dec[iThis]
                    CatOff.Name[NDone]="Off%4.4i"%NDone
                    NDone+=1


            CatOff=CatOff[CatOff.ra!=0]

        else:
            while NDone<NOff:
                # dx=np.random.rand(1)[0]*self.Radius*np.pi/180
                # dy=np.random.rand(1)[0]*self.Radius*np.pi/180
                dx=(np.random.rand(1)[0]-0.5)*2*self.Radius*np.pi/180
                dy=(np.random.rand(1)[0]-0.5)*2*self.Radius*np.pi/180
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
        self.ChanWidth = abs(tf0.getcol("CHAN_WIDTH").ravel()[0])
        tf0.close()

        times = np.unique(t0.getcol("TIME"))

        dt=times[1:]-times[:-1]
        if np.any(dt<0): stop


        t0.close()

        tField = table("%s::FIELD"%MSName, ack=False)
        if self.ra0 is None:
            self.ra0, self.dec0 = tField.getcol("PHASE_DIR").ravel() # radians!
            if self.ra0<0.: self.ra0+=2.*np.pi
        tField.close()

        self.CoordMachine = ModCoord.ClassCoordConv(self.ra0, self.dec0)

        pBAR = ProgressBar(Title="Reading metadata")
        pBAR.render(0, self.nMS)

        #for iMS, MSName in enumerate(sorted(self.ListMSName)):
        tmin,tmax=None,None
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

            tField = table("%s::FIELD"%MSName, ack=False)
            ThisRA0,ThisDEC0 = tField.getcol("PHASE_DIR").ravel() # radians!
            if ThisRA0<0.: ThisRA0+=2.*np.pi
            if (ThisRA0!=self.ra0) or (ThisDEC0!=self.dec0):
                log.print(ModColor.Str("MS %s has a different phase center - it's ok just saying"%MSName))
                log.print(ModColor.Str("MS %s has a different phase center - it's ok just saying"%MSName))
                log.print(ModColor.Str("MS %s has a different phase center - it's ok just saying"%MSName))
            tField.close()

            tf = table("%s::SPECTRAL_WINDOW"%MSName, ack=False)
            ThisTimes = np.unique(t.getcol("TIME"))
            dtBin_=np.unique(t.getcol("INTERVAL"))
            if dtBin_.size>1: stop
            dtBin = dtBin_.flat[0]


            # if not np.allclose(ThisTimes, self.times):
            #     raise ValueError("should have the same times")


            tp = table("%s::POLARIZATION"%MSName, ack=False)
            npol=tp.getcol("NUM_CORR").flat[0]
            CorrType=tp.getcol("CORR_TYPE").ravel().tolist()
            if CorrType!=[9,10,11,12]:
                raise ValueError("Pols should be XX, XY, YX, YY")
            tp.close()

            chFreq=tf.getcol("CHAN_FREQ").ravel()
            if chFreq[-1]<chFreq[0]:
                #log.print(ModColor.Str("Channels are reversed, ok I can deal with that..."))
                chSlice=slice(None,None,-1)
                RevertChans=True
            else:
                chSlice=slice(None)
                RevertChans=False


            if tmin is None:
                tmin=ThisTimes.min()
                tmax=ThisTimes.max()
            else:
                tmin=np.min([tmin,ThisTimes.min()])
                tmax=np.max([tmax,ThisTimes.max()])

            DicoMSInfos[iMS] = {"MSName": MSName,
                                "ChanFreq":   tf.getcol("CHAN_FREQ").ravel()[chSlice],  # Hz
                                "ChanWidth":  np.abs(tf.getcol("CHAN_WIDTH").ravel()), # Hz
                                "ra0dec0":  (ThisRA0,ThisDEC0) ,
                                "times":      ThisTimes,
                                "dtBin":      dtBin,
                                "npol":       npol,
                                "startTime":  Time(ThisTimes[0]/(24*3600.), format='mjd', scale='utc').isot,
                                "stopTime":   Time(ThisTimes[-1]/(24*3600.), format='mjd', scale='utc').isot,
                                "deltaTime":  (ThisTimes[-1] - ThisTimes[0])/3600., # h
                                "RevertChans": RevertChans,
                                "Readable":   True}

            if DicoMSInfos[iMS]["ChanWidth"][0] != self.ChanWidth:
                raise ValueError("should have the same chan width")
            pBAR.render(iMS+1, self.nMS)

        self.NTimesGrid=int(np.ceil((tmax-tmin)/dtBin))
        self.timesGrid=tmin+np.arange(self.NTimesGrid)*dtBin
        self.tmin=tmin
        self.tmax=tmax

        for iMS in range(self.nMS):
            if not DicoMSInfos[iMS]["Readable"]:
                print(ModColor.Str("Problem reading %s"%MSName), file=log)
                print(ModColor.Str("   %s"%DicoMSInfos[iMS]["Exception"]), file=log)


        t.close()
        tf.close()
        self.DicoMSInfos = DicoMSInfos
        self.FreqsAll    = np.array([DicoMSInfos[iMS]["ChanFreq"] for iMS in list(DicoMSInfos.keys()) if DicoMSInfos[iMS]["Readable"]])
        self.Freq_minmax = np.min(self.FreqsAll), np.max(self.FreqsAll)
        #self.NTimes      = self.times.size
        #dtArr=self.times[1:]-self.times[:-1]
        #if np.unique(dtArr).size>1:
        #    log.print(ModColor.Str("Times are not regular"))
        #dt=np.median(dtArr)



        f0, f1           = self.Freq_minmax
        self.NChan       = int((f1 - f0)/self.ChanWidth) + 1

        # Fill properties

        self.tStart = Time(tmin/(24*3600.), format='mjd', scale='utc').isot
        self.tStop  = Time(tmax/(24*3600.), format='mjd', scale='utc').isot
        self.fMin   = self.Freq_minmax[0]
        self.fMax   = self.Freq_minmax[1]

        if self.TChunkHours>0:
            TChunk_s=self.TChunkHours*3600
            TObs=self.tmax-self.tmin
            NChunks=int(np.ceil(TObs/TChunk_s))
            T0s=np.arange(NChunks)*TChunk_s
            T1s=np.arange(NChunks)*TChunk_s+TChunk_s
        else:
            T0s=np.array([0])
            T1s=np.array([1e10])
        self.T0s=T0s
        self.T1s=T1s
        #self.iCurrentMS=0
        LJob=[]
        for iMS in range(self.nMS):
            for iChunk in range(T0s.size): #[1:2]:
                T0,T1=T0s[iChunk],T1s[iChunk]
                LJob.append((iMS,iChunk))
        self.LJob=LJob


    def LoadMS(self,iJob):
        iMS,iChunk=self.LJob[iJob]
        T0,T1=self.T0s[iChunk],self.T1s[iChunk]


        if not self.DicoMSInfos[iMS]["Readable"]:
            print("Skipping [%i/%i]: %s"%(iMS+1, self.nMS, self.ListMSName[iMS]), file=log)
            return "NotRead"
        print("Reading [%i/%i] %s:%s"%(iMS+1, self.nMS, self.ListMSName[iMS],self.ColName), file=log)

        MSName=self.ListMSName[iMS]

        t = table(MSName, ack=False)

        times  = t.getcol("TIME")
        t0=self.tmin

        ind=np.where((times>=(T0+t0))&(times<(T1+t0)))[0]
        if ind.size==0:
            print("No Data in requested interval %f -> %f h time interval"%(T0/3600,T1/3600), file=log)
            return "NotRead"
        NROW=ind.size
        ROW0=ind[0]

        if ROW0!=0 or NROW!=t.nrows():
            print("  Reading chunk in %.3f -> %.3f h"%(T0/3600,T1/3600), file=log)

        nch  = self.DicoMSInfos[iMS]["ChanFreq"].size
        npol  = self.DicoMSInfos[iMS]["npol"]

        #chSlice=self.DicoMSInfos[iMS]["chSlice"]
        RevertChans=self.DicoMSInfos[iMS]["RevertChans"]

        data = np.zeros((NROW,nch,npol),np.complex64)
        t.getcolnp(self.ColName,data,ROW0,NROW)
        if RevertChans: data=data[:,::-1,:]

        if self.ModelName:
            print("  Substracting %s from %s"%(self.ModelName,self.ColName), file=log)
            model=np.zeros((NROW,nch,npol),np.complex64)
            t.getcolnp(self.ModelName,model,ROW0,NROW)
            if RevertChans: model=model[:,::-1,:]

            data-=model
            del(model)



        if self.ColWeights:
            print("  Reading weight column %s"%(self.ColWeights), file=log)
            sW=t.getcol(self.ColWeights,0,1).shape
            if len(sW)==3:
                weights=np.zeros((NROW,nch,sW[-1]),np.float32)
                t.getcolnp(self.ColWeights,weights,ROW0,NROW)
                weights=np.mean(weights,axis=-1)
            else:
                weights=np.zeros((NROW,nch),np.float32)
                t.getcolnp(self.ColWeights,weights,ROW0,NROW)


            if RevertChans: weights=weights[:,::-1]
        else:
            nrow,nch,_=data.shape
            weights=np.ones((nrow,nch),np.float32)

        flag=np.zeros((NROW,nch,npol),np.bool)
        t.getcolnp("FLAG",flag,ROW0,NROW)
        if RevertChans: flag=flag[:,::-1]


        # data[:,:,:]=0
        # data[:,0:300,0]=1
        # flag.fill(0)
        # weights.fill(1)


        times  = t.getcol("TIME",ROW0,NROW)
        A0, A1 = t.getcol("ANTENNA1",ROW0,NROW), t.getcol("ANTENNA2",ROW0,NROW)

        u, v, w = t.getcol("UVW",ROW0,NROW).T
        t.close()
        d = np.sqrt(u**2 + v**2 + w**2)
        uv0, uv1         = np.array(StrToList(self.UVRange)) * 1000
        indUV = np.where( (d<uv0)|(d>uv1) )[0]
        flag[indUV, :, :] = 1 # flag according to UV selection
        data[flag] = 0 # put down to zeros flagged visibilities

        f0, f1           = self.Freq_minmax

        # Considering another position than the phase center
        u0 = u.reshape( (-1, 1, 1) )
        v0 = v.reshape( (-1, 1, 1) )
        w0 = w.reshape( (-1, 1, 1) )
        DicoDATA = shared_dict.create("DATA_%i"%(iJob))
        DicoDATA["iMS"]=iMS
        DicoDATA["iChunk"]=iChunk
        DicoDATA["iJob"]=iJob
        DicoDATA["T0T1"]=(T0,T1)
        DicoDATA["data"]=data
        DicoDATA["weights"]=weights
        DicoDATA["flag"]=flag
        DicoDATA["times"]=times
        DicoDATA["A0"]=A0
        DicoDATA["A1"]=A1
        DicoDATA["u"]=u0
        DicoDATA["v"]=v0
        DicoDATA["w"]=w0
        DicoDATA["uniq_times"]=np.unique(DicoDATA["times"])


        if self.DoJonesCorr_kMS or self.DoJonesCorr_Beam:
            self.setJones(DicoDATA)

    def setJones(self,DicoDATA):
        from DDFacet.Data import ClassJones
        from DDFacet.Data import ClassMS
        iJob=DicoDATA["iJob"]
        iMS,iChunk=self.LJob[iJob]
        T0,T1=self.T0s[iChunk],self.T1s[iChunk]

        SolsName=self.SolsName
        if SolsName is not None and "[" in SolsName:
            SolsName=SolsName.replace("[","")
            SolsName=SolsName.replace("]","")
            SolsName=SolsName.split(",")
        GD={"Beam":{"Model":self.BeamModel,
                    "PhasedArrayMode":"A",
                    "At":"facet",
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

        ms=ClassMS.ClassMS(self.DicoMSInfos[iMS]["MSName"],GD=GD,DoReadData=False,)
        JonesMachine = ClassJones.ClassJones(GD, ms, CacheMode=False)
        JonesMachine.InitDDESols(DicoDATA)



        CutGainsMinMax=StrToList(self.options.CutGainsMinMax)
        if self.DoJonesCorr_kMS and CutGainsMinMax:
            JonesSols=DicoDATA["killMS"]["Jones"]
            G=JonesSols["Jones"]
            if CutGainsMinMax:
                c0,c1=CutGainsMinMax
                G[G>c1]=0
                G[G<c0]=0

        if self.DoJonesCorr_kMS and self.DoJonesCorr_Beam:
            DomainMachine=DDFacet.Other.ClassJonesDomains.ClassJonesDomains()
            JonesSols=DomainMachine.MergeJones(DicoDATA["killMS"]["Jones"], DicoDATA["Beam"]["Jones"])
        elif self.DoJonesCorr_kMS:
            JonesSols=DicoDATA["killMS"]["Jones"]
        elif self.DoJonesCorr_Beam:
            JonesSols=DicoDATA["Beam"]["Jones"]
        else:
            stop

        DicoJones=shared_dict.create("DicoJones_%i"%iJob)
        DicoJones["G"]=np.swapaxes(JonesSols["Jones"],1,3) # Normalize Jones matrices
        G=DicoJones["G"]
        nt,nch,na,nDir,_,_=G.shape
        DicoJones['tm']=(JonesSols["t0"]+JonesSols["t1"])/2.
        DicoJones['ra']=JonesMachine.ClusterCat['ra']
        DicoJones['dec']=JonesMachine.ClusterCat['dec']
        DicoJones['FreqDomains']=JonesSols['FreqDomains']
        DicoJones['FreqDomains_mean']=np.mean(JonesSols['FreqDomains'],axis=1)
        DicoJones['IDJones']=np.zeros((self.NDir,),np.int32)
        for iDir in range(self.NDir):
            ra=self.PosArray.ra[iDir]
            dec=self.PosArray.dec[iDir]

            #DicoJones['IDJones'][iDir]=np.argmin(AngDist(ra,DicoJones['ra'],dec,DicoJones['dec']))
            lTarget, mTarget = self.CoordMachine.radec2lm(np.array([ra]), np.array([dec]))
            lJones, mJones = self.CoordMachine.radec2lm(DicoJones['ra'], DicoJones['dec'])
            DicoJones['IDJones'][iDir]=np.argmin(np.sqrt((lTarget-lJones)**2+(mTarget-mJones)**2))



        # if self.DoJonesCorr_Beam:
        #     if not self.DoJonesCorr_kMS:
        #         RA,DEC=self.PosArray.ra,self.PosArray.dec
        #     else:
        #         RA=DicoJones_kMS['ra']
        #         DEC=DicoJones_kMS['dec']
        #     # RA,DEC=self.PosArray.ra,self.PosArray.dec
        #     JonesSols = JonesMachine.GiveBeam(np.unique(DicoDATA["times"]), quiet=True,RaDec=(RA,DEC))
        #     DicoJones_Beam['ra']=RA
        #     DicoJones_Beam['dec']=DEC
        #     DicoJones_Beam["G"]=np.swapaxes(JonesSols["Jones"],1,3) # Normalize Jones matrices
        #     DicoJones_Beam['tm']=(JonesSols["t0"]+JonesSols["t1"])/2.
        #     DicoJones_Beam['FreqDomains']=JonesSols['FreqDomains']
        #     DicoJones_Beam['FreqDomains_mean']=np.mean(JonesSols['FreqDomains'],axis=1)



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

        # from DDFacet.Data import ClassLOFARBeam
        # GD,D={},{}
        # D["PhasedArrayMode"]="A"
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


        #import DynSpecMS.testLibBeam
        T0s,T1s=self.T0s,self.T1s


        for iJob in range(len(self.LJob)):
            self.processJob(iJob)
            # iMS,iChunk=LJob[iJob]
            # rep=self.LoadMS(iJob)
            # if rep=="NotRead": continue

            # print("Making dynamic spectra...", file=log)
            # NTimes=self.NTimesGrid#self.DicoMSInfos[self.iCurrentMS]["times"].size
            # for iTime in range(NTimes):
            #     APP.runJob("Stack_SingleTime:%i_%d"%(iJob,iTime),
            #                self.Stack_SingleTime,
            #                args=(iJob,iTime,))#,serial=True)



        self.Finalise()

    def processJob(self,iJob):
        #SERIAL=True
        if iJob==0:
            self.APP.runJob("LoadMS_%i"%(iJob),
                       self.LoadMS,
                       args=(iJob,),
                       io=0)#,serial=True)

        if iJob!=len(self.LJob)-1:
            self.APP.runJob("LoadMS_%i"%(iJob+1),
                       self.LoadMS,
                       args=(iJob+1,),
                       io=0)


        # print(rep)
        rep=self.APP.awaitJobResults("LoadMS_%i"%(iJob))
        if rep=="NotRead":
            self.delShm(iJob)
            return

        # NTimes=self.NTimesGrid
        iMS,iChunk=self.LJob[iJob]
        NTimes=self.DicoMSInfos[iMS]["times"].size

        for iTime in range(NTimes):
            self.APP.runJob("Stack_SingleTime:%i_%d"%(iJob,iTime),
                       self.Stack_SingleTimeAllDir,
                       args=(iJob,iTime,))#,serial=True)

        self.APP.awaitJobResults("Stack_SingleTime:%i_*"%iJob, progress="Append MS %i"%iMS)

        self.delShm(iJob)

    def delShm(self,iJob):
        shared_dict.delDict("DATA_%i"%(iJob))
        shared_dict.delDict("DicoJones_%i"%(iJob))


    def killWorkers(self):
        print("Killing workers", file=log)
        self.APP.terminate()
        self.APP.shutdown()
        del(self.DicoGrids)
        shared_dict.delDict("Grids")
        #Multiprocessing.cleanupShm()



    def Finalise(self):

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

    # def Stack_SingleTime(self,DicoDATA,iTime):
    #     #self.DicoDATA.reload()
    #     #self.DicoGrids.reload()

    #     # for iDir in range(self.NDir):
    #     #     self.Stack_SingleTimeDir(iTime,iDir)

    #     self.Stack_SingleTimeDir(iTime)

    def Stack_SingleTimeAllDir(self,iJob,iTime):
        iMS,iChunk=self.LJob[iJob]
        DicoDATA=shared_dict.attach("DATA_%i"%(iJob))

        iMS  = DicoDATA["iMS"]
        indRow = np.where(DicoDATA["times"]==self.DicoMSInfos[iMS]["times"][iTime])[0]
        if indRow.size==0: return
        ThisTime=self.DicoMSInfos[iMS]["times"][iTime]

        nrow,nch,npol=DicoDATA["data"].shape
        indCh=np.int64(np.arange(nch)).reshape((1,nch,1))
        indPol=np.int64(np.arange(npol)).reshape((1,1,npol))
        indR=indRow.reshape((indRow.size,1,1))
        nRowOut=indRow.size
        indArr=nch*npol*np.int64(indR)+npol*np.int64(indCh)+np.int64(indPol)

        #indRow = np.where(DicoDATA["times"]>0)[0]
        #f   = DicoDATA["flag"][indRow, :, :]
        #d   = DicoDATA["data"][indRow, :, :]

        T=ClassTimeIt.ClassTimeIt("SingleTimeAllDir")
        T.disable()
        d   = np.array((DicoDATA["data"].flat[indArr.flat[:]]).reshape((nRowOut,nch,npol))).copy()
        f   = np.array((DicoDATA["flag"].flat[indArr.flat[:]]).reshape((nRowOut,nch,npol))).copy()
        T.timeit("first")

        # for i in range(10):
        #     d   = (DicoDATA["data"].flat[indArr.flat[:]]).reshape((nRowOut,nch,npol)).copy()
        #     f   = (DicoDATA["flag"].flat[indArr.flat[:]]).reshape((nRowOut,nch,npol)).copy()
        #     T.timeit("first %i"%i)

        nrow,nch,_=d.shape
        #weights   = (DicoDATA["weights"][indRow, :]).reshape((nrow,nch,1))

        indArr=nch*np.int64(indR)+np.int64(indCh)
        weights   = np.array((DicoDATA["weights"].flat[indArr.flat[:]]).reshape((nRowOut,nch,1))).copy()

        A0s = DicoDATA["A0"][indRow].copy()
        A1s = DicoDATA["A1"][indRow].copy()
        u0  = DicoDATA["u"][indRow].reshape((-1,1,1)).copy()
        v0  = DicoDATA["v"][indRow].reshape((-1,1,1)).copy()
        w0  = DicoDATA["w"][indRow].reshape((-1,1,1)).copy()
        T.timeit("second")


        chfreq=np.array(self.DicoMSInfos[iMS]["ChanFreq"].reshape((1,-1,1))).copy()
        chfreq_mean=np.mean(chfreq)
        # kk  = np.exp( -2.*np.pi*1j* f/const.c.value *(u0*l + v0*m + w0*(n-1)) ) # Phasing term
        #print iTime,iDir
        ChanFreqs=np.array(self.DicoMSInfos[iMS]["ChanFreq"][0]).copy()

        iTimeGrid=np.argmin(np.abs(self.timesGrid-self.DicoMSInfos[iMS]["times"][iTime]))

        dcorr=d.copy()
        f0, _ = self.Freq_minmax
        ich0 = int( (ChanFreqs - f0)/self.ChanWidth )
        OneMinusF=(1-f).copy()

        W=np.zeros((nRowOut,nch,npol),np.float32)
        for ipol in range(npol):
            W[:,:,ipol]=weights[:,:,0]
        W[f]=0
        Wc=W.copy()
        # weights=weights*np.ones((1,1,npol))
        # W=weights


        kk=np.zeros_like(d)
        T.timeit("third")
        for iDir in range(self.NDir):
            ra=self.PosArray.ra[iDir]
            dec=self.PosArray.dec[iDir]
            ra0,dec0=self.DicoMSInfos[iMS]["ra0dec0"]
            l, m = self.radec2lm(ra, dec,ra0,dec0)
            n  = np.sqrt(1. - l**2. - m**2.)


            T.timeit("lmn")
            kkk  = np.exp(-2.*np.pi*1j* chfreq/const.c.value *(u0*l + v0*m + w0*(n-1)) ) # Phasing term
            T.timeit("kkk")

            for ipol in range(npol):
                kk[:,:,ipol]=kkk[:,:,0]
            T.timeit("kkk copy")
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



            #DicoMSInfos      = self.DicoMSInfos

            #_,nch,_=DicoDATA["data"].shape

            dcorr[:]=d[:]
            W=Wc.copy()
            dcorr*=W
            wdcorr=np.ones(dcorr.shape,np.float64)
            #kk=kk*np.ones((1,1,npol))

            T.timeit("corr")

            if self.DoJonesCorr_kMS or self.DoJonesCorr_Beam:
                T1=ClassTimeIt.ClassTimeIt("  DoJonesCorr")
                T1.disable()
                DicoJones=shared_dict.attach("DicoJones_%i"%iJob)
                DicoJones.reload()
                T1.timeit("Load")
                tm = DicoJones['tm']
                # Time slot for the solution
                iTJones=np.argmin(np.abs(tm-ThisTime))#self.timesGrid[iTime]))

                #iDJones=np.argmin(AngDist(ra,DicoJones['ra'],dec,DicoJones['dec']))
                lJones, mJones = self.CoordMachine.radec2lm(DicoJones['ra'], DicoJones['dec'])
                iDJones=np.argmin(np.sqrt((l-lJones)**2+(m-mJones)**2))


                _,nchJones,_,_,_,_=DicoJones['G'].shape
                T1.timeit("argmin")



                for iFJones in range(nchJones):

                    nu0,nu1=DicoJones['FreqDomains'][iFJones]
                    fData=self.DicoMSInfos[iMS]["ChanFreq"].ravel()
                    indCh=np.where((fData>=nu0) & (fData<nu1))[0]

                    #iFJones=np.argmin(np.abs(chfreq_mean-DicoJones['FreqDomains_mean']))
                    # construct corrected visibilities
                    J0 = DicoJones['G'][iTJones, iFJones, A0s, iDJones, 0, 0]
                    J1 = DicoJones['G'][iTJones, iFJones, A1s, iDJones, 0, 0]

                    #JJ0=self.DicoJones['G'][iTJones, iFJones, A0s, :, 0, 0]
                    #JJ1=self.DicoJones['G'][iTJones, iFJones, A1s, :, 0, 0]
                    #indZeroJones,=np.where((JJ0=0)|(JJ1==0))


                    J0 = J0.reshape((-1, 1, 1))*np.ones((1, indCh.size, 1))
                    J1 = J1.reshape((-1, 1, 1))*np.ones((1, indCh.size, 1))
                    T1.timeit("[%i] read J0J1"%iFJones)
                    dcorr[:,indCh,:] = J0.conj() * dcorr[:,indCh,:] * J1
                    #wdcorr[:,indCh,:] *= (np.abs(J0) * np.abs(J1))**2
                    #print(iDir,iFJones,np.count_nonzero(J0==0),np.count_nonzero(J1==0))
                    #dcorr[:,indCh,:] = 1./J0 * dcorr[:,indCh,:] * 1./J1.conj()
                    W[:,indCh,:]*=(np.abs(J0) * np.abs(J1))**2
                    T1.timeit("[%i] apply "%iFJones)


                # iFJones=np.argmin(np.abs(chfreq_mean-self.DicoJones['FreqDomains_mean']))
                # # construct corrected visibilities
                # J0 = self.DicoJones['G'][iTJones, iFJones, A0s, iDJones, 0, 0]
                # J1 = self.DicoJones['G'][iTJones, iFJones, A1s, iDJones, 0, 0]
                # J0 = J0.reshape((-1, 1, 1))*np.ones((1, nch, 1))
                # J1 = J1.reshape((-1, 1, 1))*np.ones((1, nch, 1))
                # dcorr = J0.conj() * dcorr * J1

            # T.timeit("corr kMS")
            # if self.DoJonesCorr_Beam:
            #     DicoJones_Beam=shared_dict.attach("DicoJones_Beam_%i"%iJob)
            #     DicoJones_Beam.reload()
            #     tm = DicoJones_Beam['tm']
            #     # Time slot for the solution
            #     iTJones=np.argmin(np.abs(tm-self.timesGrid[iTime]))
            #     iDJones=np.argmin(AngDist(ra,DicoJones_Beam['ra'],dec,DicoJones_Beam['dec']))
            #     _,nchJones,_,_,_,_=DicoJones_Beam['G'].shape
            #     for iFJones in range(nchJones):
            #         nu0,nu1=DicoJones_Beam['FreqDomains'][iFJones]
            #         fData=self.DicoMSInfos[iMS]["ChanFreq"].ravel()
            #         indCh=np.where((fData>=nu0) & (fData<nu1))[0]
            #         #iFJones=np.argmin(np.abs(chfreq_mean-self.DicoJones_Beam['FreqDomains_mean']))
            #         # construct corrected visibilities
            #         J0 = DicoJones_Beam['G'][iTJones, iFJones, A0s, iDJones, 0, 0]
            #         J1 = DicoJones_Beam['G'][iTJones, iFJones, A1s, iDJones, 0, 0]
            #         J0 = J0.reshape((-1, 1, 1))*np.ones((1, indCh.size, 1))
            #         J1 = J1.reshape((-1, 1, 1))*np.ones((1, indCh.size, 1))
            #         dcorr[:,indCh,:] = J0.conj() * dcorr[:,indCh,:] * J1
            #         #wdcorr[:,indCh,:] *= (np.abs(J0) * np.abs(J1))**2
            #         W[:,indCh,:]*=(np.abs(J0) * np.abs(J1))**2
            #         #dcorr[:,indCh,:] = 1./J0 * dcorr[:,indCh,:] * 1./J1.conj()



            T.timeit("corr Beam")
            #ds=np.sum(d*kk, axis=0) # without Jones

            #ds = np.sum(dcorr * kk*weights, axis=0) # with Jones
            #dcorr.flat[:]*=kk.flat[:]
            #dcorr.flat[:]*=W.flat[:]
            dcorr*=kk
            #dcorr=dcorr*kk
            ds = np.sum(dcorr, axis=0) # with Jones
            #W*=wdcorr
            ws = np.sum(W, axis=0)
            w2s = np.sum(W**2, axis=0)

            # wdcorr*=W
            # dcorrs=np.sum(wdcorr, axis=0)
            # ind=np.where(ws!=0)
            # dcorrs[ind]/=ws[ind]
            # ind=np.where(dcorrs!=0)
            # ds[ind]/=dcorrs[ind]
            T.timeit("Sum")

            self.DicoGrids["GridLinPol"][iDir,ich0:ich0+nch, iTimeGrid, :] = ds
            self.DicoGrids["GridWeight"][iDir,ich0:ich0+nch, iTimeGrid, :] = np.float32(ws)
            self.DicoGrids["GridWeight2"][iDir,ich0:ich0+nch, iTimeGrid, :] = np.float32(w2s)
            T.timeit("Write")

        T.timeit("rest")



    def NormJones(self, G):
        print("  Normalising Jones matrices by the amplitude", file=log)
        G[G != 0.] /= np.abs(G[G != 0.])
        return G



    def radec2lm(self, ra, dec,ra0,dec0):
        # ra and dec must be in radians
        l = np.cos(dec) * np.sin(ra - ra0)
        m = np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(ra - ra0)
        return l, m
# =========================================================================
# =========================================================================



# Proper motion correction

from astropy.io import fits
from astropy.coordinates import SkyCoord, Distance
from astropy.time import Time
import astropy.units as u


# Assuming two things, you have read in the table and made the vairables ra, dec, pmra etc, and you the date of the observation (date_obs) in YYYY-MM-DD format.

# In terms of a pipeline I would have an if statement here that if pmra is a real value, then do the following,
# otherwise just extract at the reported ra,dec position
def ProperMotionCorrection(ra,dec,pmra,pmdec,ref_epoch,parallax,time70):
    #log.print('Performing proper motion corrections...')

    if np.isnan(pmra):
        return ra,dec

    date_obs = Time(time70/(24*3600.), format='mjd', scale='utc').iso
    if not np.isnan(parallax):
        c = SkyCoord(ra=ra * u.rad,
                     dec=dec * u.rad,
                     distance=Distance(parallax=parallax * u.mas,allow_negative=True),
                     pm_ra_cosdec=pmra * u.mas/u.yr,
                     pm_dec=pmdec * u.mas/u.yr,
                     obstime=Time(ref_epoch, format='decimalyear'))
    else:
        c = SkyCoord(ra=ra * u.rad,
                     dec=dec * u.rad,
                     #distance=Distance(parallax=parallax * u.mas,allow_negative=True),
                     pm_ra_cosdec=pmra * u.mas/u.yr,
                     pm_dec=pmdec * u.mas/u.yr,
                     obstime=Time(ref_epoch, format='decimalyear'))


    # c_lotss = SkyCoord(ra=ra_lotss * u.deg,
    #                    dec=dec_lotss * u.deg,
    #                    obstime=Time(date_obs, format='iso'))

    epoch_lotss = Time(date_obs, format='iso')

    c_gaia_to_lotss_epoch = c.apply_space_motion(epoch_lotss)
    ra1,dec1=c_gaia_to_lotss_epoch.ra.rad, c_gaia_to_lotss_epoch.dec.rad

    return ra1,dec1

    # Then to extract the correct ra and dec just do it at  c_gaia_to_lotss_epoch.ra and c_gaia_to_lotss_epoch.dec```
