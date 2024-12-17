#!/usr/bin/env python
"""
killMS, a package for calibration in radio interferometry.
Copyright (C) 2013-2017  Cyril Tasse, l'Observatoire de Paris,
SKA South Africa, Rhodes University

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
from __future__ import print_function
#!/usr/bin/env python

from builtins import str
import optparse
import pickle
#import pylab
import os
from DDFacet.Other import MyLogger
from DDFacet.Other import MyPickle
log=MyLogger.getLogger("ClassInterpol")
IdSharedMem=str(int(os.getpid()))+"."
import surveys_db

SaveName="last_InterPol.obj"

def read_options():
    desc="""Questions and suggestions: cyril.tasse@obspm.fr"""
    global options
    opt = optparse.OptionParser(usage='Usage: %prog --ms=somename.MS <options>',version='%prog version 1.0',description=desc)

    group = optparse.OptionGroup(opt, "* Data-related options", "Won't work if not specified.")
    group.add_option('--DirsDynSpecMSProds',help='SolfileIn [no default]',default=None)
    opt.add_option_group(group)


    options, arguments = opt.parse_args()
    f = open(SaveName,"wb")
    pickle.dump(options,f)


def main(options=None):
    if options==None:
        f = open(SaveName,'rb')
        options = pickle.load(f)
    
    ListDirSpec = [ l.strip() for l in open(options.DirsDynSpecMSProds).readlines() if not ".tgz" in l]

    D={}
    sdb=surveys_db.SurveysDB()
    for iFile,f in enumerate(ListDirSpec):
        OBSID=str(f.split("_L")[1])
        Di=sdb.get_observation(OBSID)
        Field=Di["field"]
        D[iFile]={}
        D[iFile]["OBSID"]=str(OBSID)
        D[iFile]["Field"]=str(Field)
    sdb.close()

    MyPickle.Save(D,"LoTSS_OBSID_vs_Field.Dico")

    for iFile in list(D.keys()):
        OBSID=D[iFile]["OBSID"]
        Field=D[iFile]["Field"]
        ss="ms2dynspec.py --imageI /databf/lofar/SURVEYS_KSP/LOTSS/DynSpecMS/IMAGES/%s/image_full_ampphase_di_m.NS_shift.int.facetRestored.fits --imageV /databf/lofar/SURVEYS_KSP/LOTSS/DynSpecMS/IMAGES/%s/image_full_low_stokesV.dirty.fits --BaseDirSpecs /databf/lofar/SURVEYS_KSP/LOTSS/DynSpecMS/DynSpecs_L%s --srclist Transient_LOTTS.csv"%(Field,Field,OBSID)
        print(ss)
        os.system(ss)

        
if __name__=="__main__":
    read_options()
    f = open(SaveName,'rb')
    options = pickle.load(f)
    
    main(options=options)



