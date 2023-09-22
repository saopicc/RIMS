import numpy as np
import AnalyseDynSpecMS.GeneDist
import pylab
import matplotlib.gridspec as gridspec
from functools import partial
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import AsyncProcessPool
import matplotlib.cm as cm
from astropy.io import fits
from SkyModel.Array import RecArrayOps
from astropy.io import ascii
import scipy.stats
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

class ClassGiveGaiaSample():
    def __init__(self,(rac_deg,decc_deg,Rad_deg),CGaia,RefCat="/data/cyril.tasse/Analyse_DataDynSpec_Jan23_TestRM/MergedCat.npz.MergeGaia.npz"):
        self.DB=np.load(RefCat,allow_pickle=1)
        C=self.DB["Cat"].view(np.recarray)
        self.C=C

        self.CGaia=CGaia
        
        ra=self.C["ra_rad"]
        ind=np.where(ra<0)[0]
        self.C["ra_rad"][ind]+=2*np.pi
        
        StatX="20x40:I"
        StatY="20x40:V"

        self.RMetric="Rn_Facet"
        self.RMetric="Rn_Global"
        
        DicoStrStatID2iStat=dict(self.DB["DicoStrStatID2iStat"][()])
        self.iStatX=DicoStrStatID2iStat[StatX]
        self.iStatY=DicoStrStatID2iStat[StatY]
        X0=C["Stats"][self.RMetric][:,self.iStatX]
        X1=C["Stats"][self.RMetric][:,self.iStatY]
        ff=C["Stats"]["flagFraction"][:,self.iStatX]
        C0=( np.logical_not(np.isnan(X0)) & np.logical_not(np.isnan(X1)) )
        C1=( (X0!=0) & (X1!=0) )
        C2=( ff<0.6 )
        C3=True#( C["RadiusCenter_deg"] < 2)
        ind=np.where(C0&C1&C2&C3)[0]
        self.C=self.C[ind]
        ind=np.where(self.C["GaiaDistance"]>0)[0]
        self.C["GaiaDistance"][ind]=0

        D=AngDist(rac_deg*np.pi/180,self.C["ra_rad"],decc_deg*np.pi/180,self.C["dec_rad"])*180/np.pi
        ind=np.where(D<Rad_deg)[0]
        self.C=self.C[ind]
        
        
        
        
        # #####################
        print(f"Sources Types in the catalog: {np.unique(self.C.SrcType)}")
        CExoP=(self.C.SrcType==b"NASA Exoplanet Archive")
        indExoP=np.where(CExoP)[0]
        NExoP=indExoP.size
        self.CExoP=self.C[indExoP]

        (rac_deg,decc_deg,Rad_deg)
        
        

    def buildRandGaiaSample(self):
        LExoP=[]
        LGaia=[]
        NExo=self.CExoP.size
        for iExo,ExoP in enumerate(self.CExoP):
            G=ExoP.G
            D=ExoP.GaiaDistance
            SrcName=ExoP.SrcName.decode('ascii')
            GaiaID=ExoP.GaiaID.decode('ascii')
            if GaiaID=="":
                print(f"[{iExo}/{NExo}] {SrcName} not Gaia ID")
                continue
            if D==0:
                print(f"[{iExo}/{NExo}] {SrcName} ({GaiaID}) D null")
                continue
            
            b_r=ExoP.b_r
            indD=np.where((self.CGaia.GaiaDistance>0.8*D) & (self.CGaia.GaiaDistance<1.2*D))[0]
            if indD.size==0:
                print(f"[{iExo}/{NExo}] no control stars for D={D}")
                continue
            CGaia_s=self.CGaia[indD]
            dG=np.abs(G-CGaia_s.G)
            db_r=np.abs(b_r-CGaia_s.b_r)
            C0=(dG<1)
            C1=(db_r<0.5)
            C3=True#(CGaia_s.ObsID==ExoP.ObsID)
            C3b=True#(CGaia_s.iFacet==ExoP.iFacet)
            dd=AngDist(ExoP.ra_rad,CGaia_s.ra_rad,ExoP.dec_rad,CGaia_s.dec_rad)*180/np.pi

            #C4=True#((dd>5/3600)&(dd<3))
            C4=((dd>5/3600))
            d_mh=np.abs(ExoP.mh-CGaia_s.mh)
            C5=True#((d_mh<0.3) & (CGaia_s.mh!=0) & (np.isnan(CGaia_s.mh)!=1))
            d_teff=np.abs(ExoP.teff-CGaia_s.teff)
            C6=True#((d_teff<800) & (CGaia_s.teff!=0) & (np.isnan(CGaia_s.teff)!=1))
            
            # print("========")
            # print("C0",np.count_nonzero(C0))
            # print("C1",np.count_nonzero(C1))
            # print("C4",np.count_nonzero(C4))
            # print("C5",np.count_nonzero(C5))

            indHR=np.where(C0&C1&C3&C3b&C4&C5&C6)[0]
            print(f"[{iExo}/{NExo}] number of control stars: {indHR.size}")
            
            if indHR.size==0: continue

            ii=int(np.random.rand(1)[0]*indHR.size)
            iii=indD[indHR[ii]]
            LGaia.append(iii)
            LExoP.append(iExo)
        
        return LExoP,LGaia
