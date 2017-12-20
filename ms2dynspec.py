#!/usr/bin/env python

__author__ = "Cyril Tasse, and Alan Loh"
#__copyright__ = "Copyright 2007, The Cogent Project"
__credits__ = ["Cyril Tasse", "Alan Loh"]
#__license__ = "GPL"
__version__ = "1.0.1"
#__maintainer__ = "Rob Knight"
#__email__ = "rob@spot.colorado.edu"
#__status__ = "Production"
SaveFile = "last_dynspec.obj"

"""
=========================================================================
                                DESCRIPTION
    Blablabla
    Modif 
    
    Example:
    python ms2dynspec.py --msfile /usr/data/ --data CORRECTED --model PREDICT --sols killms.npz --srclist SRCPOS.txt --rad 10

-------------------------------------------------------------------------
                                TO DO
- convertSrclist: only keep (RA, Dec) wich are within the field -- DONE
- convertSrclist: add some other ~random positions on wich to compute dynamic spectra for comparison
- stokes computation: CHECK correct I Q U V computation! -- DONE
=========================================================================
"""

import sys
import os
import argparse
from pyrap.tables import table
from astropy.time import Time
from astropy import units as uni
from astropy.io import fits
from astropy import coordinates as coord
from astropy import constants as const
import numpy as np
import glob, os
import pylab
from DDFacet.Other import MyPickle
import logo
logo.PrintLogo()
from ClassDynSpecMS import ClassDynSpecMS
import ClassSaveResults
from DDFacet.Data.ClassMS import expandMSList
from DDFacet.Other import ModColor
from DDFacet.Other import progressbar

# =========================================================================
# =========================================================================
def angSep(ra1, dec1, ra2, dec2):
    """ Find the angular separation of two sources (ra# dec# in deg) in deg
        (Stolen from the LOFAR scripts), works --> compared with astropy (A. Loh)
    """
    b = (np.pi / 2) - np.radians(dec1)
    c = (np.pi / 2) - np.radians(dec2)
    temp = (np.cos(b) * np.cos(c)) + (np.sin(b) * np.sin(c) * np.cos(np.radians(ra1 - ra2)))
    if abs(temp) > 1.0:
        temp = 1.0 * cmp(temp, 0)
    return np.degrees(np.arccos(temp))



def main(args=None, messages=[]):
    if args is None:
        args = MyPickle.Load(SaveFile)
    
    MSList=expandMSList(args.ms)
    MSList=[mstuple[0] for mstuple in MSList]

    D = ClassDynSpecMS(ListMSName=MSList, 
                       ColName=args.data, ModelName=args.model, 
                       SolsName=args.sols, 
                       UVRange=args.uv,
                       FileCoords=args.srclist,
                       Radius=args.rad,
                       NOff=args.noff,
                       Image=args.image)

    if D.NDirSelected==0:
        return
    D.StackAll()

    SaveMachine=ClassSaveResults.ClassSaveResults(D)
    SaveMachine.WriteFits()
    SaveMachine.PlotSpec()
    SaveMachine.tarDirectory()

# =========================================================================
# =========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ms", type=str, help="Name of MS file / directory", required=True)
    parser.add_argument("--data", type=str, default="CORRECTED", help="Name of DATA column", required=True)
    parser.add_argument("--model", type=str, help="Name of MODEL column", required=True)
    parser.add_argument("--sols", type=str, help="Jones solutions", required=True)
    parser.add_argument("--srclist", type=str, default=None, help="List of targets --> 'source_name ra dec'")
    parser.add_argument("--rad", type=float, default=3., help="Radius of the field", required=False)
    parser.add_argument("--noff", type=float, default=-1, help="Number of off sources. -1 means twice as much as there are sources in the catalog", required=False)
    parser.add_argument("--LogBoring", type=int, default=0, help="Boring?", required=False)
    parser.add_argument("--image", type=str, default=None, help="Survey image to plot", required=False)
    parser.add_argument("--uv", type=list, default=[1., 1000.], help="UV range in km [UVmin, UVmax]", required=False)
    args = parser.parse_args()

    MyPickle.Save(args, SaveFile)

    ModColor.silent = progressbar.ProgressBar.silent = args.LogBoring

    main(args)
