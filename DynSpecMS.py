from pyrap.tables import table
import numpy as np
import glob
import pylab

# test git

def testAll():

    lll=['/data/loh/LOFAR/V830Tau/LC7015/results-L567729/L567729',
         '/data/loh/LOFAR/V830Tau/LC7015/results-L573623/L573623',
         '/data/loh/LOFAR/V830Tau/LC7015/results-L573687/L573687',
         '/data/loh/LOFAR/V830Tau/LC7015/results-L573613/L573613']
    
    for ll in lll:
        ll=sorted(glob.glob("%s*"%ll))#[0:10]
        D=DynSpecMS(ll)
        D.StackAll()
        D.PlotSpec()


def test():

    
    ll=sorted(glob.glob("/data/loh/LOFAR/V830Tau/LC7015/results-L573613/L573613*"))[0:1]
    D=DynSpecMS(ll)
    D.StackAll()
    return D.PlotSpec()
    

class DynSpecMS():
    def __init__(self,ListMSName,ColName="DATA",UVRange=[1.,1000.]):
        self.ListMSName=ListMSName
        self.ColName=ColName
        self.OutName=self.ListMSName[0].split("/")[-1].split("_")[0]
        self.UVRange=UVRange
        self.ReadMSInfos()

    def ReadMSInfos(self):
        print "Reading data"
        DicoMSInfos={}
        t0=table(self.ListMSName[0],ack=False)
        tf0=table("%s::SPECTRAL_WINDOW"%self.ListMSName[0],ack=False)
        self.ChanWidth=tf0.getcol("CHAN_WIDTH").ravel()[0]
        tf0.close()
        self.times=np.unique(t0.getcol("TIME"))
        t0.close()
        
        for iMS,MSName in enumerate(sorted(self.ListMSName)):
            print MSName
            try:
                t=table(MSName,ack=False)
            except:
                DicoMSInfos[iMS]={"Readable":False}
                print "problem reading %s"%MSName
                continue

            tf=table("%s::SPECTRAL_WINDOW"%MSName,ack=False)
            ThisTimes=np.unique(t.getcol("TIME"))
            if not np.allclose(ThisTimes,self.times):
                raise ValueError("should have the same times")
            DicoMSInfos[iMS]={"MSName":MSName,
                              "ChanFreq":tf.getcol("CHAN_FREQ").ravel(),
                              "ChanWidth":tf.getcol("CHAN_WIDTH").ravel(),
                              "times":ThisTimes,
                              "Readable":True}
            if DicoMSInfos[iMS]["ChanWidth"][0]!=self.ChanWidth:
                raise ValueError("should have the same chan width")

        t.close()
        tf.close()
        self.nMS=len(DicoMSInfos)
        self.DicoMSInfos=DicoMSInfos
        self.FreqsAll=[DicoMSInfos[iMS]["ChanFreq"] for iMS in DicoMSInfos.keys() if DicoMSInfos[iMS]["Readable"]]
        self.Freq_minmax=np.min(self.FreqsAll),np.max(self.FreqsAll)
        self.NTimes=self.times.size
        f0,f1=self.Freq_minmax
        self.NChan=int((f1-f0)/self.ChanWidth)+1
        self.IGrid=np.zeros((self.NChan,self.NTimes,4),np.complex128)
        
    def StackAll(self):
        print "Stacking data"

        f0,_=self.Freq_minmax

        self.WeightsGrid=np.zeros_like(self.IGrid)
        DicoMSInfos=self.DicoMSInfos
        uv0,uv1=np.array(self.UVRange)*1000

        for iMS,MSName in enumerate(sorted(self.ListMSName)):
            print "%i/%i"%(iMS,self.nMS)
            if not DicoMSInfos[iMS]["Readable"]: continue

            t=table(MSName,ack=False)
            data=t.getcol(self.ColName)-t.getcol("PREDICT_KMS")
            #data=t.getcol("PREDICT_KMS")

            flag=t.getcol("FLAG")
            times=t.getcol("TIME")
            u,v,w=t.getcol("UVW").T
            d=np.sqrt(u**2+v**2+w**2)
            indUV=np.where((d<uv0)|(d>uv1))[0]
            flag[indUV,:,:]=1

            data[flag]=0

            ich0=int((DicoMSInfos[iMS]["ChanFreq"][0]-f0)/self.ChanWidth)
            nch=DicoMSInfos[iMS]["ChanFreq"].size
            

            for iTime in range(self.NTimes):
                indRow=np.where(times==self.times[iTime])[0]
                f=flag[indRow,:,:]
                d=data[indRow,:,:]

                
                ds=np.sum(d,axis=0)
                ws=np.sum(1-f,axis=0)
                self.IGrid[ich0:ich0+nch,iTime,:]=ds
                self.WeightsGrid[ich0:ich0+nch,iTime,:]=ws

        W=self.WeightsGrid
        G=self.IGrid
        W[W==0]=1
        Gn=G/W

        GOut=np.zeros_like(self.IGrid)
        GOut[:,:,0]=Gn[:,:,0]+Gn[:,:,3]
        GOut[:,:,1]=Gn[:,:,0]-Gn[:,:,3]
        GOut[:,:,2]=Gn[:,:,1]+Gn[:,:,2]
        GOut[:,:,3]=1j*(Gn[:,:,1]-Gn[:,:,2])
        self.GOut=GOut

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
        fig.savefig("DynSpec_%s.png"%self.OutName)
        return Gn
