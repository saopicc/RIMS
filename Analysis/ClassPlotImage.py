from astropy.time import Time
from astropy import units as uni
from astropy.io import fits
from astropy.wcs import WCS
from astropy import coordinates as coord
from astropy import constants as const
import numpy as np
import glob, os
#import pylab
#from DDFacet.Other import logger
#log=logger.getLogger("ClassSaveResults")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as pylab
# from DDFacet.Other.progressbar import ProgressBar
# from pyrap.images import image
from astropy.time import Time

def GiveMAD(X):
    return np.median(np.abs(X-np.median(X)))

class ClassPlotImage():

    def __init__(self,ImageV):
        self.ImageV=ImageV
        #self.imV=image(self.DynSpecMS.ImageV)
        #self.ImageVData=self.imV.getdata()[0,pol]
        
        f = fits.open(ImageV)[0]
        self.w = WCS(f.header)
        self.headerv=f.header
        self.ImageVData=f.data[0]


        
    def Plot(self,fig,gs,rac,decc,BoxArcSec=30,pol=0):
        headerv = fits.getheader(self.ImageV) # TO BE MODIFIED
        datav   = self.ImageVData[pol,:, :] # TO BE MODIFIED
        #f,p,ra,dec=self.imV.toworld([0,0,0,0]) # self.im TO BE MODIFIED
        #_,_,xc,yc=self.imV.topixel([f,p,self.DynSpecMS.PosArray.dec[iDir], self.DynSpecMS.PosArray.ra[iDir]])
        
        yc,xc,_,_=self.w.wcs_world2pix(rac,decc,0,0,1)
        yc=yc[()]
        xc=xc[()]


        yc,xc=int(xc),int(yc)
        wcs    = WCS(self.headerv).celestial
        CDEL   = wcs.wcs.cdelt
        pos_ra_pix, pos_dec_pix = wcs.wcs_world2pix(np.degrees(rac), np.degrees(decc), 1)
        nn=self.ImageVData.shape[-1]
        #boxv = int(box / np.abs(wcs.wcs.cdelt[0]) * np.abs(CDEL[0]))
        boxv=int(abs((BoxArcSec/3600.)/wcs.wcs.cdelt[0]))
        def giveBounded(x):
            x=np.max([0,x])
            return np.min([x,nn-1])
        x0=giveBounded(xc-boxv)
        x1=giveBounded(xc+boxv)
        y0=giveBounded(yc-boxv)
        y1=giveBounded(yc+boxv)
        DataBoxed=datav[y0:y1,x0:x1]


        
        FluxV=datav[yc,xc]
        sigFluxV=GiveMAD(DataBoxed)
        
        newra_cen, newdec_cen = wcs.wcs_pix2world( (x1+x0)/2., (y1+y0)/2., 1)
        wcs.wcs.crpix  = [ DataBoxed.shape[1]/2., DataBoxed.shape[0]/2. ] # update the WCS object
        wcs.wcs.crval = [ newra_cen, newdec_cen ]
        ax1 = fig.add_subplot(gs, projection=wcs)
        
        if DataBoxed.size>boxv:
            std=GiveMAD(DataBoxed)
            vMin, vMax    = (-5.*std, 30*std)

            bigfont   = 8
            smallfont = 6
            
            im = pylab.imshow(DataBoxed, interpolation="nearest", cmap='bone_r', aspect="auto", vmin=vMin, vmax=vMax,
                              origin='lower', rasterized=True)
            cbar = pylab.colorbar()
            ax1.set_xlabel(r'RA (J2000)')
            raax = ax1.coords[0]
            raax.set_major_formatter('hh:mm:ss')
            raax.set_ticklabel(size=smallfont)
            ax1.set_ylabel(r'Dec (J2000)')
            
            decax = ax1.coords[1]
            decax.set_major_formatter('dd:mm:ss')
            decax.set_ticklabel(size=smallfont)
            ax1.autoscale(False)
            
            cbar.set_label(r'Flux density (mJy)', fontsize=bigfont, horizontalalignment='center')
            cbar.ax.tick_params(labelsize=smallfont)
            pylab.setp(ax1.get_xticklabels(), rotation='horizontal', fontsize=smallfont)
            pylab.setp(ax1.get_yticklabels(), rotation='horizontal', fontsize=smallfont)
            ra_cen, dec_cen = wcs.wcs_world2pix(np.degrees(rac), np.degrees(decc), 1)
            pylab.plot(DataBoxed.shape[1]/2., DataBoxed.shape[0]/2., 'o', markerfacecolor='none', markeredgecolor='red', markersize=bigfont) # plot a circle at the target
            pylab.text(DataBoxed.shape[0]*0.9, DataBoxed.shape[1]*0.9, 'V', horizontalalignment='center', verticalalignment='center', fontsize=bigfont+2) 
