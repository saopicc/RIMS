from pyrap.tables import table
from astropy.time import Time
from astropy import units as uni
from astropy.io import fits
from astropy import coordinates as coord
from astropy import constants as const
import numpy as np
import glob, os
import pylab

# deuxieme
# test git

def testAll():

    lll=['/data/loh/LOFAR/V830Tau/LC7015/results-L567729/L567729',
         '/data/loh/LOFAR/V830Tau/LC7015/results-L573623/L573623',
         '/data/loh/LOFAR/V830Tau/LC7015/results-L573687/L573687',
         '/data/loh/LOFAR/V830Tau/LC7015/results-L573613/L573613']
    #lll=['/data/loh/LOFAR/V830Tau/LC7015/results-L567729/L567729']
    
    for ll in lll:
        ll=sorted(glob.glob("%s*"%ll))#[0:10]
        D=DynSpecMS(ll)
        #D.StackAll() # center
        #D.StackAll(ra=68.42, dec=24.58) # no source
        #D.StackAll(ra=68.274, dec=24.558) # aretefact galaxy


        # --- Final ---
        #D.StackAll() # V830 Tau (TEST_DynSpec_L573613_raNone_decNone.fits, TEST_DynSpec_L573687_raNone_decNone.fits, TEST_DynSpec_L573623_raNone_decNone.fits, TEST_DynSpec_L567729_raNone_decNone.fits)
        #D.StackAll(ra=68.30209, dec=24.552526) # OFF1 close to V830tau facet (DynSpec_L573613_ra68.30209_dec24.552526_OFF1.fits)
        D.StackAll(ra=68.344491, dec=22.951881) # OFF2

        D.WriteFits()
        #D.PlotSpec()
    return

def test():

    
    ll=sorted(glob.glob("/data/loh/LOFAR/V830Tau/LC7015/results-L573613/L573613*"))[0:1]
    D=DynSpecMS(ll)
    D.StackAll()
    D.WriteFits()
    return D.PlotSpec()
    

class DynSpecMS():
    def __init__(self, ListMSName, ColName="DATA", UVRange=[1.,1000.], Sols='killMS.ALL_SOLS_1km_KAFCA.p.sols.npz'):
        self.ListMSName = ListMSName
        self.ColName    = ColName
        self.OutName    = self.ListMSName[0].split("/")[-1].split("_")[0]
        self.UVRange    = UVRange
        self.Sols       = Sols
        self.tStart     = None
        self.tStop      = None
        self.fMin       = None
        self.fMax       = None
        self.ra0        = None # phase center (rad)
        self.dec0       = None
        self.ra         = None
        self.dec        = None
        self.ReadMSInfos()

    def ReadMSInfos(self):
        print "Reading data"
        DicoMSInfos={}
        t0  = table(self.ListMSName[0], ack=False)
        tf0 = table("%s::SPECTRAL_WINDOW"%self.ListMSName[0],ack=False)
        self.ChanWidth = tf0.getcol("CHAN_WIDTH").ravel()[0]
        tf0.close()
        self.times     = np.unique(t0.getcol("TIME"))
        t0.close()
        
        for iMS, MSName in enumerate(sorted(self.ListMSName)):
            print MSName
            try:
                t = table(MSName, ack=False)
            except:
                DicoMSInfos[iMS] = {"Readable":False}
                print "problem reading %s"%MSName
                continue

            # Extract Jones matrices to apply to the visibilities
            try:
                if self.Sols[-3:] != 'npz':
                    raise Exception("Solutions should be a .npz file")
                JonesSols = np.load("%s/%s"%(MSName, self.Sols))
                # Normalize Jones matrices
                JonesSols = self.NormJones(JonesSols)
            except:
                print "problem reading solutions %s/%s"%(MSName, self.Sols)
                JonesSols = None

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
                            "Readable":   True,
                            "JonesSols":  JonesSols}
            if DicoMSInfos[iMS]["ChanWidth"][0] != self.ChanWidth:
                raise ValueError("should have the same chan width")

        t.close()
        tf.close()
        self.nMS         = len(DicoMSInfos)
        self.DicoMSInfos = DicoMSInfos
        self.FreqsAll    = [DicoMSInfos[iMS]["ChanFreq"] for iMS in DicoMSInfos.keys() if DicoMSInfos[iMS]["Readable"]]
        self.Freq_minmax = np.min(self.FreqsAll), np.max(self.FreqsAll)
        self.NTimes      = self.times.size
        f0, f1           = self.Freq_minmax
        self.NChan       = int((f1 - f0)/self.ChanWidth) + 1
        self.IGrid       = np.zeros((self.NChan, self.NTimes, 4), np.complex128)

        
        
    def StackAll(self, ra=None, dec=None):   
        print "Stacking data"
        

        f0, _ = self.Freq_minmax
        self.ra  = ra
        self.dec = dec

        self.WeightsGrid = np.zeros_like(self.IGrid)
        DicoMSInfos      = self.DicoMSInfos
        uv0, uv1         = np.array(self.UVRange) * 1000

        for iMS, MSName in enumerate(sorted(self.ListMSName)):
            print "%i/%i"%(iMS, self.nMS)
            if not DicoMSInfos[iMS]["Readable"]: continue

            t      = table(MSName, ack=False)
            tField = table("%s::FIELD"%MSName)
            self.ra0, self.dec0 = tField.getcol("PHASE_DIR").ravel() # radians!
            phaseCent = coord.SkyCoord(self.ra0, self.dec0, unit=uni.rad)
            if (self.ra is not None) and (self.dec is not None):
                phaseCent = coord.SkyCoord(self.ra, self.dec, unit=uni.deg) # not phase center but still...
            data   = t.getcol(self.ColName) - t.getcol("PREDICT_KMS") # Residuals
            #data   = t.getcol("PREDICT_KMS") # Model
            #data   = t.getcol(self.ColName) # data

            flag   = t.getcol("FLAG")
            times  = t.getcol("TIME")
            A0, A1 = t.getcol("ANTENNA1"), t.getcol("ANTENNA2")

            u, v, w = t.getcol("UVW").T
            d = np.sqrt(u**2 + v**2 + w**2)
            indUV = np.where( (d<uv0)|(d>uv1) )[0]
            flag[indUV, :, :] = 1 # flag according to UV selection

            data[flag] = 0 # put down to zeros flagged visibilities

            ich0 = int( (DicoMSInfos[iMS]["ChanFreq"][0] - f0)/self.ChanWidth )
            nch  = DicoMSInfos[iMS]["ChanFreq"].size

            # ------------------------------------------------------------------------------------------------
            #                               Pixel direction to sum visibilities
            # ------------------------------------------------------------------------------------------------
            if ra is None and dec is None:
                # considering centrak pixel for the dynamic spectrum computation
                ra  = np.degrees(self.ra0)
                dec = np.degrees(self.dec0)
                
            # Considering another position than the phase center
            u0 = u.reshape( (-1, 1, 1) )
            v0 = v.reshape( (-1, 1, 1) )
            w0 = w.reshape( (-1, 1, 1) )
            f  = DicoMSInfos[iMS]["ChanFreq"].reshape( (1, nch, 1) )
            l, m = self.radec2lm(np.radians(ra), np.radians(dec))
            n = np.sqrt(1. - l**2. - m**2.)
            k = np.exp( -2.*np.pi*1j* f/const.c.value *(u0*l + v0*m + w0*(n-1)) ) # check le - (verifier apres soustraction, mettre ra dec d'une source forte pour verifier)
            # ------------------------------------------------------------------------------------------------
             
           
            
            for iTime in range(self.NTimes):
                indRow = np.where(times==self.times[iTime])[0]
                f   = flag[indRow, :, :]
                d   = data[indRow, :, :]
                A0s = A0[indRow]
                A1s = A1[indRow]
                kk  = k[indRow, :, :]

                # ------------------------------------------------------------------------------------------------
                #                                          Apply Jones matrices
                # ------------------------------------------------------------------------------------------------
                if DicoMSInfos[iMS]["JonesSols"] is not None:
                    self.J = DicoMSInfos[iMS]["JonesSols"]['G']
                    t0 = DicoMSInfos[iMS]["JonesSols"]['t0']
                    t1 = DicoMSInfos[iMS]["JonesSols"]['t1']
                    
                    # Time slot for the solution
                    iTJones = np.where( t0 <= self.times[iTime] )[0][-1]
                    # Facet used
                    posFacets = coord.SkyCoord(DicoMSInfos[iMS]["JonesSols"]['ra'], DicoMSInfos[iMS]["JonesSols"]['dec'], unit=uni.rad)
                    iDJones = phaseCent.separation(posFacets).deg.argmin()                
                    # construct corrected visibilities
                    J0 = self.J[iTJones, 0, A0s, iDJones, 0, 0]
                    J1 = self.J[iTJones, 0, A1s, iDJones, 0, 0]

                    J0 = J0.reshape((-1, 1, 1))*np.ones((1, nch, 1))
                    J1 = J1.reshape((-1, 1, 1))*np.ones((1, nch, 1))

                    dcorr = J0.conj() * d * J1
                # ------------------------------------------------------------------------------------------------

                #ds=np.sum(d*kk, axis=0) # without Jones
                ds = np.sum(dcorr * kk, axis=0) # with Jones
                ws = np.sum(1-f, axis=0)
                self.IGrid[ich0:ich0+nch, iTime, :]       = ds
                self.WeightsGrid[ich0:ich0+nch, iTime, :] = ws

            # Fill properties
            self.tStart = DicoMSInfos[iMS]["startTime"]
            self.tStop  = DicoMSInfos[iMS]["stopTime"] 
            self.fMin   = self.Freq_minmax[0]
            self.fMax   = self.Freq_minmax[1]

        W = self.WeightsGrid
        G = self.IGrid
        W[W == 0] = 1
        Gn = G/W # 

        GOut=np.zeros_like(self.IGrid)
        GOut[:, :, 0] = Gn[:, :, 0] + Gn[:, :, 3]
        GOut[:, :, 1] = Gn[:, :, 0] - Gn[:, :, 3]
        GOut[:, :, 2] = Gn[:, :, 1] + Gn[:, :, 2]
        GOut[:, :, 3] = 1j*(Gn[:, :, 1] - Gn[:, :, 2])
        self.GOut = GOut

    def NormJones(self, jMatrices):
        """ Normalisation of Jones Matrices
        """
        t0   = jMatrices['Sols']['t0']
        t1   = jMatrices['Sols']['t1']
        sols = jMatrices['Sols']['G']
        ra   = jMatrices['ClusterCat']['ra']
        dec  = jMatrices['ClusterCat']['dec']
        for iTime in xrange(sols.shape[0]):
            for iFreq in xrange(sols.shape[1]):
                for iAnt in xrange(sols.shape[2]):
                    for iDir in xrange(sols.shape[3]):
                        # 2D matrix [[xx, xy], [yx, yy]]
                        sols[iTime, iFreq, iAnt, iDir] /= np.linalg.norm(sols[iTime, iFreq, iAnt, iDir])
        return {'G':sols, 't0':t0, 't1':t1, 'ra':ra, 'dec':dec}     
    def WriteFits(self):
        """ Store the dynamic spectrum in a FITS file
        """
        fitsname = "DynSpec_{}_ra{}_dec{}_OFF2.fits".format(self.OutName, self.ra, self.dec)

        # Create the fits file
        prihdr  = fits.Header() 
        prihdr.set('DATE-CRE', Time.now().iso.split()[0], 'Date of file generation')
        prihdr.set('CHAN-WID', self.ChanWidth, 'Frequency channel width')
        prihdr.set('FRQ-MIN', self.fMin, 'Minimal frequency')
        prihdr.set('FRQ-MAX', self.fMax, 'Maximal frequency')
        prihdr.set('OBS-STAR', self.tStart, 'Observation start date')
        prihdr.set('OBS-STOP', self.tStop, 'Observation end date')
        hdus    = fits.PrimaryHDU(header=prihdr) # hdu table that will be filled
        hdus.writeto(fitsname, clobber=True)

        label = ["I", "Q", "U", "V"]
        hdus = fits.open(fitsname)
        for ipol in range(4):
            Gn = self.GOut[:, :, ipol].real
            hdr   = fits.Header()
            hdr.set('STOKES', label[ipol], '')
            hdus.append( fits.ImageHDU(data=Gn, header=hdr, name=label[ipol]) )
        hdulist = fits.HDUList(hdus)
        hdulist.writeto(fitsname, clobber=True)
        hdulist.close()
        print("{} written".format(fitsname))
        return

    def PlotSpec(self):


        fig=pylab.figure(1,figsize=(15,8))
        label=["I","Q","U","V"]
        pylab.clf()
        for ipol in range(4):
            Gn=self.GOut[:,:,ipol].T.real
            v0,v1=np.percentile(Gn.ravel(),[10.,90.])
            pylab.subplot(2,2,ipol+1)
            pylab.imshow(Gn,interpolation="nearest",aspect="auto",vmin=3*v0,vmax=10*v1)
            pylab.title(label[ipol])
            pylab.colorbar()
            pylab.ylabel("Time bin")
            pylab.xlabel("Freq bin")
        pylab.tight_layout()
        pylab.draw()
        pylab.show(False)
        fig.savefig("DynSpec_{}_ra{}_dec{}.png".format(self.OutName, self.ra, self.dec))
        return Gn


    def radec2lm(self, ra, dec):
        # ra and dec must be in radians
        l = np.cos(dec) * np.sin(ra - self.ra0)
        m = np.sin(dec) * np.cos(self.dec0) - np.cos(dec) * np.sin(self.dec0) * np.cos(ra - self.ra0)
        return l, m


