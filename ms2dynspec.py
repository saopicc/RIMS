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
from DynSpecMS import DynSpecMS
import ClassSaveResults
   
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


def checkArgs(args):
    """ Check that the arguments are correct
    """
    if (not os.path.isfile(args.ms)) and (not os.path.exists(args.ms)): # TO DO: if it is a directory where ae stored MS files...
        raise Exception("\t=== MS file/directory '{}' not found ===".format(args.ms))
    else:
        if args.ms[-2:].upper() == 'MS':
            t = table(args.ms, ack=False)
            args.ms = [args.ms]
        else:
            args.ms = sorted( glob.glob("%s/*"%os.path.abspath(args.ms)) )
            t = table(args.ms[0], ack=False)

    if args.data not in t.colnames():
        raise Exception("\t=== DATA column '{}' not found ===".format(args.data))

    if args.model not in t.colnames():
        raise Exception("\t=== MODEL column '{}' not found ===".format(args.model))

    if args.sols[-3:] != 'npz':
        raise Exception("\t=== Solution '{}' should be a .npz file ===".format(args.sols))

    # if args.srclist is not None:
    #     if not os.path.isfile(args.srclist):
    #          raise Exception("\t=== Source list file '{}'' not found ===".format(args.srclist))
    #     args.srclist = convertSrclist(srclist=args.srclist, ms=args.ms[0], fieldradius=args.rad)
    #     if len(args.srclist.keys()) == 0:
    #         raise Exception("\t=== No source in the list ===")
    # else:
    #     # Stop the pgrm
    #     raise Exception("\t=== No need to run the script if no source list ===")

    t.close()
    return # Everything's ok

# =========================================================================
# =========================================================================

def main(args=None, messages=[]):
    if args is None:
        args = MyPickle.Load(SaveFile)
    
    D = DynSpecMS(ListMSName=args.ms, 
                  ColName=args.data, ModelName=args.model, 
                  SolsName=args.sols, 
                  UVRange=args.uv,
                  FileCoords=args.srclist,
                  Radius=args.rad,
                  NOff=args.noff)
    if D.NDir==0:
        return
    D.StackAll()

    SaveMachine=ClassSaveResults.ClassSaveResults(D)
    SaveMachine.WriteFits()
    SaveMachine.PlotSpec()


# =========================================================================
# =========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ms", type=str, help="Name of MS file / directory", required=True)
    parser.add_argument("--data", type=str, default="CORRECTED", help="Name of DATA column", required=True)
    parser.add_argument("--model", type=str, help="Name of MODEL column", required=True)
    parser.add_argument("--sols", type=str, help="Jones solutions", required=True)
    parser.add_argument("--srclist", type=str, default=None, help="List of targets --> 'source_name ra dec'", required=True)
    parser.add_argument("--rad", type=float, default=3., help="Radius of the field", required=False)
    parser.add_argument("--noff", type=float, default=-1, help="Number of off sources. -1 means twice as much as there are sources in the catalog", required=False)
    parser.add_argument("--uv", type=list, default=[1., 1000.], help="UV range in km [UVmin, UVmax]", required=False)
    args = parser.parse_args()
    checkArgs(args) # Verify that everything is correct before launching the dynamic spectrum computation
    MyPickle.Save(args, SaveFile)

    main(args)
