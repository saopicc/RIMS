from astropy.time import Time
from astropy import units as uni
from astropy.io import fits
from astropy import coordinates as coord
from astropy import constants as const
import numpy as np
import glob, os
#import pylab
from DDFacet.Other import MyLogger
log=MyLogger.getLogger("ClassSaveResults")
from DDFacet.ToolsDir.rad2hmsdms import rad2hmsdms
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as pylab
from DDFacet.Other.progressbar import ProgressBar


class ClassSaveResults():
    def __init__(self, DynSpecMS):
        self.DynSpecMS=DynSpecMS
        self.DIRNAME="DynSpecs_%s"%self.DynSpecMS.OutName

        os.system("rm -rf %s"%self.DIRNAME)
        os.system("mkdir -p %s/TARGET"%self.DIRNAME)
        os.system("mkdir -p %s/OFF"%self.DIRNAME)
        #os.system("mkdir -p %s/PNG"%self.DIRNAME)

    def tarDirectory(self):
        os.system("tar -zcvf %s.tgz %s"%(self.DIRNAME,self.DIRNAME))


    def WriteFits(self):
        for iDir in range(self.DynSpecMS.NDir):
            self.WriteFitsThisDir(iDir)

        self.WriteFitsThisDir(0,Weight=True)

    def GiveSubDir(self,Type):
        SubDir="OFF"
        if Type!="Off": SubDir="TARGET"
        return SubDir

    def WriteFitsThisDir(self,iDir,Weight=False):
        """ Store the dynamic spectrum in a FITS file
        """
        ra,dec=self.DynSpecMS.PosArray.ra[iDir],self.DynSpecMS.PosArray.dec[iDir]
        
        strRA=rad2hmsdms(ra,Type="ra").replace(" ",":")
        strDEC=rad2hmsdms(dec,Type="dec").replace(" ",":")

        fitsname = "%s/%s/%s_%s_%s.fits"%(self.DIRNAME,self.GiveSubDir(self.DynSpecMS.PosArray.Type[iDir]),self.DynSpecMS.OutName, strRA, strDEC)
        if Weight:
            fitsname = "%s/%s.fits"%(self.DIRNAME,"Weights")
            

        # Create the fits file
        prihdr  = fits.Header() 
        prihdr.set('DATE-CRE', Time.now().iso.split()[0], 'Date of file generation')
        prihdr.set('OBSID', self.DynSpecMS.OutName, 'LOFAR Observation ID')
        prihdr.set('CHAN-WID', self.DynSpecMS.ChanWidth, 'Frequency channel width')
        prihdr.set('FRQ-MIN', self.DynSpecMS.fMin, 'Minimal frequency')
        prihdr.set('FRQ-MAX', self.DynSpecMS.fMax, 'Maximal frequency')
        prihdr.set('OBS-STAR', self.DynSpecMS.tStart, 'Observation start date')
        prihdr.set('OBS-STOP', self.DynSpecMS.tStop, 'Observation end date')
        prihdr.set('RA_RAD', ra, 'Pixel right assention')
        prihdr.set('DEC_RAD', dec, 'Pixel right declination')
        hdus    = fits.PrimaryHDU(header=prihdr) # hdu table that will be filled
        hdus.writeto(fitsname, clobber=True)

        label = ["I", "Q", "U", "V"]
        PolID = [0,1,2,3]
        hdus = fits.open(fitsname)
        for iLabel in range(len(label)):
            if Weight:
                Gn = self.DynSpecMS.DicoGrids["GridWeight"][iDir,:, :, PolID[iLabel]].real
            else:
                Gn = self.DynSpecMS.GOut[iDir,:, :, PolID[iLabel]].real
            hdr   = fits.Header()
            hdr.set('STOKES', label[iLabel], '')
            hdr.set('RMS', np.std(Gn), 'r.m.s. of the data')
            hdr.set('MEAN', np.mean(Gn), 'Mean of the data')
            hdr.set('MEDIAN', np.median(Gn), 'Median of the data')
            hdus.append( fits.ImageHDU(data=Gn, header=hdr, name=label[iLabel]) )
        hdulist = fits.HDUList(hdus)
        print>>log,"  --> Writting %s"%fitsname
        hdulist.writeto(fitsname, clobber=True)#)#, overwrite=True)
        hdulist.close()


    def PlotSpec(self):
        
        pdfname = "%s/%s.pdf"%(self.DIRNAME,self.DynSpecMS.OutName)
        print>>log,"Making pdf ovserview: %s"%pdfname
        pBAR = ProgressBar(Title="Making pages")
        NPages=self.DynSpecMS.NDir#Selected
        pBAR.render(0, NPages)

        with PdfPages(pdfname) as pdf:
            for iDir in range(self.DynSpecMS.NDir):
                fig = pylab.figure(1, figsize=(15, 8))
                pBAR.render(iDir+1, NPages)
                if self.DynSpecMS.PosArray.Type[iDir]=="Off": continue
                self.PlotSpecSingleDir(iDir)
                pdf.savefig()
                pylab.close()

    def PlotSpecSingleDir(self,iDir=0):
        label = ["I", "Q", "U", "V"]
        #pylab.clf()
        
        ra,dec=self.DynSpecMS.PosArray.ra[iDir],self.DynSpecMS.PosArray.dec[iDir]
        strRA=rad2hmsdms(ra,Type="ra").replace(" ",":")
        strDEC=rad2hmsdms(dec,Type="dec").replace(" ",":")

        for ipol in range(4):
            Gn = self.DynSpecMS.GOut[iDir,:, :, ipol].T.real
            sig = np.median(np.abs(Gn))
            mean = np.median(Gn)
            pylab.subplot(2, 2, ipol+1)
            pylab.imshow(Gn, interpolation="nearest", aspect="auto", vmin=mean-3*sig, vmax=mean+10*sig)
            pylab.title(label[ipol])
            pylab.colorbar()
            pylab.ylabel("Time bin")
            pylab.xlabel("Freq bin")
            pylab.suptitle("Name: %s, Type: %s"%(self.DynSpecMS.PosArray.Name[iDir],self.DynSpecMS.PosArray.Type[iDir]))


