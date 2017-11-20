from astropy.time import Time
from astropy import units as uni
from astropy.io import fits
from astropy.wcs import WCS
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
        deltaT = (Time(self.DynSpecMS.times[1]/(24*3600.), format='mjd', scale='utc') - Time(self.DynSpecMS.times[0]/(24*3600.), format='mjd', scale='utc')).sec
        prihdr.set('TIME-WID', deltaT, 'Time bin width (sec)')
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
                self.fig = pylab.figure(1, figsize=(15, 15))
                pBAR.render(iDir+1, NPages)
                if self.DynSpecMS.PosArray.Type[iDir]=="Off": continue
                self.PlotSpecSingleDir(iDir)
                pdf.savefig(bbox_inches='tight')
                pylab.close()

    def PlotSpecSingleDir(self, iDir=0):
        label = ["I", "Q", "U", "V"]
        pylab.clf()

        # Figure properties
        bigfont   = 8
        smallfont = 6
        ra, dec = self.DynSpecMS.PosArray.ra[iDir],self.DynSpecMS.PosArray.dec[iDir]
        strRA  = rad2hmsdms(ra, Type="ra").replace(" ", ":")
        strDEC = rad2hmsdms(dec, Type="dec").replace(" ", ":")
        freqs = self.DynSpecMS.FreqsAll.ravel() * 1.e-6 # in MHz
        t0 = Time(self.DynSpecMS.times[0]/(24*3600.), format='mjd', scale='utc')
        t1 = Time(self.DynSpecMS.times[-1]/(24*3600.), format='mjd', scale='utc')
        times = np.linspace(0, (t1-t0).sec/60., num=self.DynSpecMS.GOut[0, :, :, 0].shape[1], endpoint=True)
        image  = self.DynSpecMS.Image

        if (image is None) | (not os.path.isfile(image)):
            # Just plot a series of dynamic spectra
            for ipol in range(4):
                # Gn = self.DynSpecMS.GOut[iDir,:, :, ipol].T.real
                # sig = np.median(np.abs(Gn))
                # mean = np.median(Gn)
                # pylab.subplot(2, 2, ipol+1)
                # pylab.imshow(Gn, interpolation="nearest", aspect="auto", vmin=mean-3*sig, vmax=mean+10*sig)
                # pylab.title(label[ipol])
                # pylab.colorbar()
                # pylab.ylabel("Time bin")
                # pylab.xlabel("Freq bin")
                Gn = self.DynSpecMS.GOut[iDir,:, :, ipol].real
                sig  = np.std(np.abs(Gn))
                mean = np.median(Gn)
                ax1 = pylab.subplot(2, 2, ipol+1)
                spec = pylab.pcolormesh(times, freqs, Gn, cmap='bone_r', vmin=mean-3*sig, vmax=mean+10*sig, rasterized=True)
                ax1.axis('tight')
                cbar = pylab.colorbar()
                cbar.ax.tick_params(labelsize=6)
                pylab.text(times[-1]-0.1*(times[-1]-times[0]), freqs[-1]-0.1*(freqs[-1]-freqs[0]), label[ipol], horizontalalignment='center', verticalalignment='center', fontsize=bigfont)
                if ipol==2 or ipol==3:
                    pylab.xlabel("Time (min since %s)"%(t0.iso), fontsize=bigfont)
                pylab.ylabel("Frequency (MHz)", fontsize=bigfont)
                pylab.setp(ax1.get_xticklabels(), rotation='horizontal', fontsize=smallfont)
                pylab.setp(ax1.get_yticklabels(), rotation='horizontal', fontsize=smallfont)
        else:
            # Plot the survey image and the dynamic spectra series
            # ---- Dynamic spectra I  ----
            axspec = pylab.subplot2grid((5, 2), (2, 0), colspan=2)
            Gn   = self.DynSpecMS.GOut[iDir,:, :, 0].real
            sig  = np.std(np.abs(Gn))
            mean = np.median(Gn)
            #spec = pylab.pcolormesh(times, freqs, Gn, cmap='bone_r', vmin=mean-3*sig, vmax=mean+10*sig, rasterized=True)
            spec = pylab.imshow(Gn, cmap='bone_r', vmin=mean-3*sig, vmax=mean+10*sig, extent=(times[0],times[-1],self.DynSpecMS.fMin,self.DynSpecMS.fMax)) 
            axspec.axis('tight')
            cbar = pylab.colorbar(fraction=0.046, pad=0.01)
            cbar.ax.tick_params(labelsize=smallfont)
            cbar.set_label(r'Flux density (Jy)', fontsize=8, horizontalalignment='center') 
            pylab.text(times[-1]-0.02*(times[-1]-times[0]), freqs[-1]-0.1*(freqs[-1]-freqs[0]), 'I', horizontalalignment='center', verticalalignment='center', fontsize=bigfont+2)
            pylab.xlabel("Time (min since %s)"%(t0.iso), fontsize=bigfont)
            pylab.ylabel("Frequency (MHz)", fontsize=bigfont)
            pylab.setp(axspec.get_xticklabels(), rotation='horizontal', fontsize=smallfont)
            pylab.setp(axspec.get_yticklabels(), rotation='horizontal', fontsize=smallfont)
            # ---- Dynamic spectra L  ----
            axspec = pylab.subplot2grid((5, 2), (3, 0), colspan=2)
            Gn   = np.sqrt(self.DynSpecMS.GOut[iDir,:, :, 1].real**2. + self.DynSpecMS.GOut[iDir,:, :, 2].real**2.)
            sig  = np.std(np.abs(Gn))
            mean = np.median(Gn) 
            #spec = pylab.pcolormesh(times, freqs, Gn, cmap='bone_r', vmin=0, vmax=mean+10*sig, rasterized=True)
            spec = pylab.imshow(Gn, cmap='bone_r', vmin=mean-3*sig, vmax=mean+10*sig, extent=(times[0],times[-1],self.DynSpecMS.fMin,self.DynSpecMS.fMax)) 
            axspec.axis('tight')
            cbar = pylab.colorbar(fraction=0.046, pad=0.01)
            cbar.ax.tick_params(labelsize=smallfont)
            cbar.set_label(r'Flux density (Jy)', fontsize=8, horizontalalignment='center') 
            pylab.text(times[-1]-0.02*(times[-1]-times[0]), freqs[-1]-0.1*(freqs[-1]-freqs[0]), 'L', horizontalalignment='center', verticalalignment='center', fontsize=bigfont+2)
            pylab.xlabel("Time (min since %s)"%(t0.iso), fontsize=bigfont)
            pylab.ylabel("Frequency (MHz)", fontsize=bigfont)
            pylab.setp(axspec.get_xticklabels(), rotation='horizontal', fontsize=smallfont)
            pylab.setp(axspec.get_yticklabels(), rotation='horizontal', fontsize=smallfont)
            # ---- Dynamic spectra V  ----
            axspec = pylab.subplot2grid((5, 2), (4, 0), colspan=2)
            Gn   = self.DynSpecMS.GOut[iDir,:, :, 3].real
            sig  = np.std(np.abs(Gn))
            mean = np.median(Gn) 
            #spec = pylab.pcolormesh(times, freqs, Gn, cmap='bone_r', vmin=mean-3*sig, vmax=mean+10*sig, rasterized=True)
            spec = pylab.imshow(Gn, cmap='bone_r', vmin=mean-3*sig, vmax=mean+10*sig, extent=(times[0],times[-1],self.DynSpecMS.fMin,self.DynSpecMS.fMax)) 
            axspec.axis('tight')
            cbar = pylab.colorbar(fraction=0.046, pad=0.01)
            cbar.ax.tick_params(labelsize=smallfont)
            cbar.set_label(r'Flux density (Jy)', fontsize=8, horizontalalignment='center') 
            pylab.text(times[-1]-0.02*(times[-1]-times[0]), freqs[-1]-0.1*(freqs[-1]-freqs[0]), 'V', horizontalalignment='center', verticalalignment='center', fontsize=bigfont+2)
            pylab.xlabel("Time (min since %s)"%(t0.iso), fontsize=bigfont)
            pylab.ylabel("Frequency (MHz)", fontsize=bigfont)
            pylab.setp(axspec.get_xticklabels(), rotation='horizontal', fontsize=smallfont)
            pylab.setp(axspec.get_yticklabels(), rotation='horizontal', fontsize=smallfont)

            # ---- Plot mean vs time  ----
            ax2 = pylab.subplot2grid((5, 2), (0, 1))
            Gn_i = self.DynSpecMS.GOut[iDir,:, :, 0].real
            meantime = np.mean(Gn_i, axis=0)
            stdtime  = np.std(Gn_i, axis=0)
            ax2.fill_between(times, meantime-stdtime, meantime+stdtime, facecolor='#B6CAC8', edgecolor='none', zorder=-10)
            pylab.plot(times, meantime, color='black')
            pylab.axhline(y=0, color='black', linestyle=':')
            pylab.xlabel("Time (min since %s)"%(t0.iso), fontsize=bigfont)
            pylab.ylabel("Mean (Stokes I)", fontsize=bigfont)
            pylab.setp(ax2.get_xticklabels(), rotation='horizontal', fontsize=smallfont)
            pylab.setp(ax2.get_yticklabels(), rotation='horizontal', fontsize=smallfont)
            ymin, vv = np.percentile((meantime-stdtime).ravel(), [5, 95])
            vv, ymax = np.percentile((meantime+stdtime).ravel(), [5, 95])
            ax2.set_ylim([ymin, ymax])
            ax2.set_xlim([times[0], times[-1]])

            # ---- Plot mean vs frequency  ----
            ax3 = pylab.subplot2grid((5, 2), (1, 1))
            meanfreq = np.mean(Gn_i, axis=1)
            stdfreq = np.std(Gn_i, axis=1)
            ax3.fill_between(freqs.ravel(), meanfreq-stdfreq, meanfreq+stdfreq, facecolor='#B6CAC8', edgecolor='none', zorder=-10)
            ax3.plot(freqs.ravel(), meanfreq, color='black')
            ax3.axhline(y=0, color='black', linestyle=':')
            pylab.xlabel("Frequency (MHz)", fontsize=bigfont)
            pylab.ylabel("Mean (Stokes I)", fontsize=bigfont)
            pylab.setp(ax3.get_xticklabels(), rotation='horizontal', fontsize=smallfont)
            pylab.setp(ax3.get_yticklabels(), rotation='horizontal', fontsize=smallfont)
            ymin, vv = np.percentile((meanfreq-stdfreq).ravel(), [5, 95])
            vv, ymax = np.percentile((meanfreq+stdfreq).ravel(), [5, 95])
            ax3.set_ylim([ymin,ymax])
            ax3.set_xlim([freqs.ravel()[0], freqs.ravel()[-1]])

            # ---- Image ----
            npix = 1000
            header = fits.getheader(image)
            data   = np.squeeze(fits.getdata(image, ext=0)) # A VERIFIER
            wcs    = WCS(header).celestial
            cenpixx, cenpixy = wcs.wcs_world2pix(self.DynSpecMS.PosArray.ra[iDir], self.DynSpecMS.PosArray.dec[iDir], 1) # get central pixels
            data = 1.e3 * data[int(cenpixx-npix/2):int(cenpixx+npix/2), int(cenpixy-npix/2):int(cenpixy+npix/2)] # resize the image 
            wcs.wcs.crpix = [npix/2, npix/2] # update the WCS object
            median, stdev = (np.median(data), np.std(data))
            vMin, vMax    = (median - 1*stdev, median + 5*stdev)
            ax1 = pylab.subplot2grid((5, 2), (0, 0), rowspan=2, projection=wcs)
            im = pylab.imshow(data, interpolation="nearest", cmap='bone_r', aspect="auto", vmin=vMin, vmax=vMax, origin='lower')
            cbar = pylab.colorbar()#(fraction=0.046*2., pad=0.01*4.)
            ax1.set_xlabel(r'RA (J2000)', fontsize=bigfont)
            raax = ax1.coords[0]
            raax.set_major_formatter('hh:mm:ss')
            raax.set_ticklabel(size=smallfont)
            ax1.set_ylabel(r'Dec (J2000)', fontsize=bigfont)
            decax = ax1.coords[1]
            decax.set_major_formatter('dd:mm:ss')
            decax.set_ticklabel(size=smallfont)
            ax1.autoscale(False)
            pylab.plot(npix/2, npix/2, 'o', markerfacecolor='none', markeredgecolor='red', markersize=bigfont) # plot a circle at the target
            cbar.set_label(r'Flux density (mJy)', fontsize=bigfont, horizontalalignment='center')
            cbar.ax.tick_params(labelsize=smallfont)
            pylab.setp(ax1.get_xticklabels(), rotation='horizontal', fontsize=smallfont)
            pylab.setp(ax1.get_yticklabels(), rotation='horizontal', fontsize=smallfont)

        #pylab.subplots_adjust(wspace=0.15, hspace=0.30)
        pylab.figtext(x=0.5, y=0.92, s="Name: %s, Type: %s, RA: %s, Dec: %s"%(self.DynSpecMS.PosArray.Name[iDir], self.DynSpecMS.PosArray.Type[iDir], self.DynSpecMS.PosArray.ra[iDir], self.DynSpecMS.PosArray.dec[iDir]), fontsize=bigfont+2, horizontalalignment='center', verticalalignment='bottom')
        #pylab.suptitle("Name: %s, Type: %s, RA: %s, Dec: %s"%(self.DynSpecMS.PosArray.Name[iDir], self.DynSpecMS.PosArray.Type[iDir], self.DynSpecMS.PosArray.ra[iDir], self.DynSpecMS.PosArray.dec[iDir]))



