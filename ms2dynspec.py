#!/usr/bin/env python

from __future__ import division
from __future__ import absolute_import
from past.builtins import cmp
__author__ = "Cyril Tasse, and Alan Loh"
__credits__ = ["Cyril Tasse", "Alan Loh"]
from dynspecms_version import version
__version__ = version()
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
from distutils.spawn import find_executable
import matplotlib
matplotlib.use('Agg')
from matplotlib import rc
fontsize=12
rc('font',**{'family':'serif','serif':['Times'],'size':fontsize})
if find_executable("latex") is not None:
    rc('text', usetex=True)

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
logo.PrintLogo(__version__)
from ClassDynSpecMS import ClassDynSpecMS
import ClassSaveResults
from DDFacet.Data.ClassMS import expandMSList
from DDFacet.Other import ModColor
from DDFacet.Other import progressbar

# # ##############################
# # Catch numpy warning
# np.seterr(all='raise')
# import warnings
# warnings.filterwarnings('error')
# #with warnings.catch_warnings():
# #    warnings.filterwarnings('error')
# # ##############################

# =========================================================================
# =========================================================================
def angSep(ra1, dec1, ra2, dec2):
    """ Find the angular separation of two sources (ra# dec# in deg) in deg
        (Stolen from the LOFAR scripts), works --> compared with astropy (A. Loh)
    """
    b = np.pi/2 - np.radians(dec1)
    c = np.pi/2 - np.radians(dec2)
    temp = (np.cos(b) * np.cos(c)) + (np.sin(b) * np.sin(c) * np.cos(np.radians(ra1 - ra2)))
    if abs(temp) > 1.0:
        temp = 1.0 * cmp(temp, 0)
    return np.degrees(np.arccos(temp))



def main(args=None, messages=[]):
    if args is None:
        args = MyPickle.Load(SaveFile)

    MSList=None
    if args.ms:
        MSList=expandMSList(args.ms)
        MSList=[mstuple[0] for mstuple in MSList]

    D = ClassDynSpecMS(ListMSName=MSList, 
                       ColName=args.data, ModelName=args.model, 
                       SolsName=args.sols,
                       ColWeights=args.WeightCol,
                       UVRange=args.uv,
                       FileCoords=args.srclist,
                       Radius=args.rad,
                       NOff=args.noff,
                       ImageI=args.imageI,
                       ImageV=args.imageV,
                       SolsDir=args.SolsDir,NCPU=args.NCPU,
                       BaseDirSpecs=args.BaseDirSpecs,
                       BeamModel=args.BeamModel,
                       BeamNBand=args.BeamNBand)

    if D.NDirSelected==0:
        return

    if D.Mode=="Spec": D.StackAll()

    SaveMachine=ClassSaveResults.ClassSaveResults(D)
    if D.Mode=="Spec":
        SaveMachine.WriteFits()
        SaveMachine.PlotSpec()
        SaveMachine.SaveCatalog()
        SaveMachine.tarDirectory()
    else:
        SaveMachine.SaveCatalog()
        SaveMachine.PlotSpec(Prefix="_replot")


        

# =========================================================================
# =========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ms", type=str, help="Name of MS file / directory", required=False)
    parser.add_argument("--data", type=str, default="CORRECTED", help="Name of DATA column", required=False)
    parser.add_argument("--WeightCol", type=str, default=None, help="Name of weights column to be taken into account", required=False)
    parser.add_argument("--model", type=str, help="Name of MODEL column",default="")#, required=True)
    parser.add_argument("--sols", type=str, help="Jones solutions",default="")
    parser.add_argument("--srclist", type=str, default="Transient_LOTTS.csv", help="List of targets --> 'source_name ra dec'")
    parser.add_argument("--rad", type=float, default=3., help="Radius of the field", required=False)
    parser.add_argument("--noff", type=int, default=-1, help="Number of off sources. -1 means twice as much as there are sources in the catalog", required=False)
    parser.add_argument("--LogBoring", type=int, default=0, help="Boring?", required=False)
    parser.add_argument("--imageI", type=str, default=None, help="Survey image to plot", required=False)
    parser.add_argument("--imageV", type=str, default=None, help="Survey image to plot", required=False)
    parser.add_argument("--BaseDirSpecs", type=str, default=None, help="Path to the precomputed specs", required=False)
    parser.add_argument("--uv", type=list, default=[1., 1000.], help="UV range in km [UVmin, UVmax]", required=False)
    parser.add_argument("--SolsDir", type=str, default="", help="Base directory for the DDE solutions", required=False)
    parser.add_argument("--NCPU", type=int, default=0, help="NCPU", required=False)
    parser.add_argument("--BeamModel", type=str, default=None, help="Beam Model to be used", required=False)
    parser.add_argument("--BeamNBand", type=int, default=1, help="Number of channels in the Beam Jones matrix", required=False)

    args = parser.parse_args()

    MyPickle.Save(args, SaveFile)

    ModColor.silent = progressbar.ProgressBar.silent = args.LogBoring

    main(args)
