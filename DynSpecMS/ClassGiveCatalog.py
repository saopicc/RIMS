import numpy as np
from DDFacet.Other import logger
from SkyModel.Sky import ModRegFile
log=logger.getLogger("DynSpecMS")
from astropy.io import fits
from astropy.wcs import WCS
import random
from astropy.io import ascii
import astropy.coordinates as coord
import astropy.units as u


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

class ClassGiveCatalog():
    def __init__(self,options,ra0,dec0,Radius,FileCoords):
        self.options=options
        self.ra0=ra0
        self.dec0=dec0
        self.Radius=Radius
        self.FileCoords=FileCoords

        
    def giveCat(self,SubSet=None):
        FileCoords=self.FileCoords
        dtype=[('Name','S200'),("ra",np.float64),("dec",np.float64),('Type','S200')]
        
        self.DoProperMotionCorr=False
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
        elif self.options.UseGaiaDB is not None:
            from astroquery.gaia import Gaia
            rac_deg,decc_deg=self.ra0*180/np.pi, self.dec0*180/np.pi
            Radius_deg=self.Radius
            Dmax,NMax=self.options.UseGaiaDB.split(",")
            Dmax,NMax=float(Dmax),int(NMax)
            Parallax_min=1./(Dmax*1e-3)
            query=f"""SELECT TOP 10000 gaia_source.designation,gaia_source.source_id,gaia_source.ref_epoch,gaia_source.ra,gaia_source.dec,gaia_source.parallax,gaia_source.pmra,gaia_source.pmdec
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
                   ("pmra",np.float64),("pmdec",np.float64),("ref_epoch",np.float64),("parallax",np.float64),
                   ('Type','S200')]
            for r in result:
                l.append((r['DESIGNATION'],r['ra'],r['dec'],r['pmra'],r['pmdec'],r['ref_epoch'],r['parallax'],b"Gaia DR3"))
            self.PosArray=np.asarray(l,dtype=dtype)

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

            self.DoProperMotionCorr=True
            if self.PosArray.size>NMax:
                ind=np.int64(np.random.rand(NMax)*self.PosArray.size)
                self.PosArray=self.PosArray[ind]
                
            
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
                if ".reg" in FileCoords:
                    R=ModRegFile.Reg2Np(FileCoords)
                    R="/data/cyril.tasse/DataDynSpec_Jan23/P151+52/image_full_ampphase_di_m.NS.BrightSel.reg"
                    stop
                else:
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
            self.DoProperMotionCorr=True
        elif FileCoords is not None:
            tbl = ascii.read(FileCoords)
            ra = coord.Angle(tbl["ra"], unit=u.hour)
            dec = coord.Angle(tbl["dec"], unit=u.degree)
            self.PosArray=np.zeros((len(tbl),),dtype=dtype)
            self.PosArray["ra"]=ra.degree
            self.PosArray["dec"]=dec.degree
            self.PosArray["Name"][:]=tbl["Name"][:]

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
        self.NOrig=self.PosArray.Name.shape[0]
        Dist=AngDist(self.ra0,self.PosArray.ra,self.dec0,self.PosArray.dec)
        ind=np.where(Dist<(Radius*np.pi/180))[0]
        self.PosArray=self.PosArray[ind]
        
        print("Created an array with %i records" % self.PosArray.size, file=log)

        if SubSet is not None:
            random.seed(42)
            i,n=SubSet
            ind=random.sample(range(0, self.PosArray.size), self.PosArray.size)
            ii=np.int64(np.linspace(0,self.PosArray.size+1,n+1))
            L=np.sort(ind[ii[i]:ii[i+1]])
            self.PosArray=self.PosArray[L]
            print("Selected %i objects for subset %i/%i" % (self.PosArray.size,i+1,n), file=log)
            
        return self.PosArray
