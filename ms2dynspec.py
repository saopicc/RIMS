#!/usr/bin/env python

from __future__ import division
from __future__ import absolute_import
import matplotlib
matplotlib.use('Agg')
from past.builtins import cmp
__author__ = "Cyril Tasse, and Alan Loh"
__credits__ = ["Cyril Tasse", "Alan Loh"]
from dynspecms_version import version
from DDFacet.Other import logger
log=logger.getLogger("ms2dynspec")
from DDFacet.Other import ModColor
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
from matplotlib import rc
fontsize=12
rc('font',**{'family':'serif','serif':['Times'],'size':fontsize})
if find_executable("latex") is not None:
    rc('text', usetex=True)
from DDFacet.Other import Multiprocessing

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
#from ClassDynSpecMS import logo
#logo.PrintLogo(__version__)
from ClassDynSpecMS import ClassDynSpecMS
from ClassSaveResults import ClassSaveResults
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

    if args.SplitNonContiguous:
        DT={}
        for MSName in MSList:
            t=table(MSName,ack=False)
            Times=np.unique(t.getcol("TIME"))
            T=(Times.min(),Times.max())
            if T not in DT.keys():
                DT[T]=[MSName]
            else:
                DT[T].append(MSName)
            t.close()
        if len(DT)>1:
            log.print(ModColor.Str("FOUND %i time periods"%len(DT)))
    else:
        DT={0:MSList}

    for k in DT.keys():
        MSList=DT[k]
        D = ClassDynSpecMS(ListMSName=MSList,
                           ColName=args.data, ModelName=args.model,
                           SolsName=args.sols,
                           TChunkHours=args.TChunkHours,
                           ColWeights=args.WeightCol,
                           UVRange=args.uv,
                           FileCoords=args.srclist,
                           Radius=args.rad,
                           NOff=args.noff,
                           DicoFacet=args.DicoFacet,
                           ImageI=args.imageI,
                           ImageV=args.imageV,
                           SolsDir=args.SolsDir,NCPU=args.NCPU,
                           BaseDirSpecs=args.BaseDirSpecs,
                           BeamModel=args.BeamModel,
                           BeamNBand=args.BeamNBand,
                           SourceCatOff_FluxMean=args.SourceCatOff_FluxMean,
                           SourceCatOff_dFluxMean=args.SourceCatOff_dFluxMean,
                           SourceCatOff=args.SourceCatOff,
                           options=args)

        if D.NDirSelected==0:
            return

        if D.Mode=="Spec": D.StackAll()

        SaveMachine=ClassSaveResults.ClassSaveResults(D,DIRNAME=args.OutDirName)
        if D.Mode=="Spec":
            SaveMachine.WriteFits()
            if args.SavePDF:
                SaveMachine.PlotSpec()
            SaveMachine.SaveCatalog()
            if args.DoTar: SaveMachine.tarDirectory()
        else:
            SaveMachine.SaveCatalog()
            SaveMachine.PlotSpec(Prefix="_replot")
        D.killWorkers()
    Multiprocessing.cleanupShm()


# =========================================================================
# =========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ms", type=str, help="Name of MS file / directory", required=False)
    parser.add_argument("--data", type=str, default="CORRECTED", help="Name of DATA column", required=False)
    parser.add_argument("--TChunkHours", type=float, default=0., help="Chunk size in hours", required=False)

    parser.add_argument("--WeightCol", type=str, default=None, help="Name of weights column to be taken into account", required=False)
    parser.add_argument("--model", type=str, help="Name of MODEL column",default="")#, required=True)
    parser.add_argument("--sols", type=str, help="Jones solutions",default="")
    parser.add_argument("--srclist", type=str, default="", help="List of targets --> 'source_name ra dec'")
    parser.add_argument("--FitsCatalog", type=str, default="", help="FITS catalog. List of targets --> Name,ra,dec,pmra,pmdec,ref_epoch,parallax,Type")
    parser.add_argument("--rad", type=float, default=3., help="Radius of the field", required=False)
    parser.add_argument("--noff", type=int, default=-1, help="Number of off sources. -1 means twice as much as there are sources in the catalog", required=False)
    parser.add_argument("--nMinOffPerFacet", type=int, default=5, help="Minimum of off sources per facet if DicoFacet is specified.", required=False)
    parser.add_argument("--DicoFacet", type=str, default="", help="DDFacet DicoFacet file.", required=False)
    parser.add_argument("--LogBoring", type=int, default=0, help="Boring?", required=False)
    parser.add_argument("--imageI", type=str, default=None, help="Survey image to plot", required=False)
    parser.add_argument("--imageV", type=str, default=None, help="Survey image to plot", required=False)
    parser.add_argument("--BaseDirSpecs", type=str, default=None, help="Path to the precomputed specs", required=False)
    parser.add_argument("--uv", type=str, default=[1., 1000.], help="UV range in km [UVmin, UVmax]", required=False)
    parser.add_argument("--SolsDir", type=str, default="", help="Base directory for the DDE solutions", required=False)
    parser.add_argument("--CutGainsMinMax", type=str, default="None", help="Cut Jones min,max", required=False)
    parser.add_argument("--SplitNonContiguous", type=int, default=1, help="Split non time-contiguous MSs", required=False)
    parser.add_argument("--UseLoTSSDB", type=int, default=0, help="Use LoTSS DB for target list", required=False)
    parser.add_argument("--UseGaiaDB", type=str, default=None, help="Use Gaia DB for target list", required=False)
    parser.add_argument("--DoTar", type=int, default=1, help="Tar final products", required=False)

    parser.add_argument("--NCPU", type=int, default=0, help="NCPU", required=False)
    parser.add_argument("--BeamModel", type=str, default=None, help="Beam Model to be used", required=False)
    parser.add_argument("--BeamNBand", type=int, default=1, help="Number of channels in the Beam Jones matrix", required=False)
    parser.add_argument("--OutDirName", type=str, default="MSName", help="Name of the output directory name", required=False)
    parser.add_argument("--SavePDF", type=int, default=0, help="Save PDF", required=False)
    parser.add_argument("--SourceCatOff", type=str, default="", help="Read the code", required=False)
    parser.add_argument("--SourceCatOff_FluxMean", type=float, default=0, help="Read the code", required=False)
    parser.add_argument("--SourceCatOff_dFluxMean", type=float, default=0, help="Read the code", required=False)

    args = parser.parse_args()

    MyPickle.Save(args, SaveFile)

    ModColor.silent = progressbar.ProgressBar.silent = args.LogBoring

    main(args)
