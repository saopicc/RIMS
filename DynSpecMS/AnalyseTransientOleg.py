#!/usr/bin/env python

from Analysis.ClassCovariance import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import os
import sys

from numpy import fft

def fd2p(f):
    """Converts Faraday dispersion function to P(K). \phi=0 and k=0 is at pixel N/2"""
    return fft.fftshift(fft.ifft(fft.ifftshift(f)))

def p2fd(p):
    """Converts P(k) to Faraday dispersion function. \phi=0 and k=0 is at pixel N/2"""
    return fft.fftshift(fft.fft(fft.ifftshift(p)))



def testBruce():
    
    # runAllDir(Patern="/home/ctasse/TestDynSpecMS/TestWeights",
    #          SaveDir="/home/ctasse/TestDynSpecMS/PNG_DEEP_Jy_briggs",
    #          UseLoTSSDB=False)
    
    # runAllDir(Patern="/home/ctasse/TestDynSpecMS/TestWeights",
    #          SaveDir="/home/ctasse/TestDynSpecMS/PNG_DEEP_Jy_4Panel",
    #          UseLoTSSDB=False)
    
    # runAllDir(Patern="/home/ctasse/TestDynSpecMS/TestWeights_Natural",
    #          SaveDir="/home/ctasse/TestDynSpecMS/PNG_DEEP_Jy_Natural_4Panel",
    #          UseLoTSSDB=False)

    # runAllDir(Patern="TestWeights_NewObsTestWeights_NewControl_Briggs",
    #          SaveDir="/home/ctasse/TestDynSpecMS/PNG_DEEP_Jy_NewControl_Briggs",
    #          UseLoTSSDB=False)
    
    # runAllDir(Patern="TestWeights_NewObs",
    #          SaveDir="/home/ctasse/TestDynSpecMS/PNG_DEEP_Jy_NewObs_Natural",
    #          UseLoTSSDB=False)

    #runAllDir(Patern="Pealed_NewObs_Control2",
    #         SaveDir="/home/ctasse/TestDynSpecMS/PNG_Pealed_NewObs_Control2",
    #         UseLoTSSDB=False)
    
    # runAllDir(Patern="FullJupiter_NoSubTarget_SourceOff",
    #           SaveDir=None,
    #           UseLoTSSDB=False)
    
    runAllDir(Patern="FullJupiter_NoSubTarget_SourceOff",
              SaveDir=None,
              UseLoTSSDB=False)
    
    # runAllDir(Patern="/home/ctasse/TestDynSpecMS/TestWeights_WhereChan",
    #           SaveDir="/home/ctasse/TestDynSpecMS/PNG_DEEP_whereChan",
    #           UseLoTSSDB=False)


def runAllDir(Patern="/data/cyril.tasse/DataDynSpec_May21/*/DynSpecs_*",SaveDir=None,UseLoTSSDB=False,LTimes=None):

    if SaveDir is None:
        SaveDir="%s/PNG_%s"%(os.getcwd(),Patern)
        
    L=glob.glob(Patern)
    print(L)
    if UseLoTSSDB:
        with SurveysDB() as sdb:
            sdb.cur.execute('UNLOCK TABLES')
            sdb.cur.execute('select * from spectra')
            result=sdb.cur.fetchall()
        DB={}
        
        for t in result:
            F=t["filename"].split("/")[-1]
            DB[F]=t
    else:
        
        #DB={"1608538564_20:09:36.800_-20:26:46.000.fits":{"filename":"/data/cyril.tasse/TestDynSpecMS/DynSpecs_1608538564/TARGET/1608538564_20:09:36.800_-20:26:46.000.fits","type":"Oleg"}}
        DB={}
        LTarget=[]
        for DirName in L:
            LTarget+=glob.glob("%s/TARGET/*.fits"%DirName)
            
        for f in LTarget:
            F=f.split("/")[-1]
            DB[F]={"filename":f,"type":"NoDB"}
            
    
    
    
    
    #L=["/data/cyril.tasse/DataDynSpec_May21/P156+42/DynSpecs_L352758"]
    


    
    DBOut=[]

        
    # #tp = ThreadPool(50)
    # for iDir,BaseDir in enumerate(L):
    #     tp.apply_async(doRunDir, (BaseDir,))
    # tp.close()
    # tp.join()

    # Jobs=[(d,DB) for d in L]#[0:10]
    # with Pool(90) as p:
    #     print(p.map(doRunDir, Jobs))
    # return


    for iDir,BaseDir in enumerate(L):
        
        print("========================== [%i / %i]"%(iDir,len(L)))
        print(BaseDir)

        L_CRD=[]
        Lpol=["I","Q","U","V"]
        #Lpol=["I"]
        InMask=None
        for pol in Lpol:
            CRD0=ClassRunDir(BaseDir=BaseDir,DB=DB,pol=pol,SaveDir=SaveDir,InMask=InMask)
            CRD0.runDir()
            InMask=CRD0.Mask
            L_CRD.append(CRD0)
        # CRD.Plot()
        
        #CRD1=ClassRunDir(BaseDir=BaseDir,DB=DB,pol="U",SaveDir=SaveDir)
        #CRD1.runDir()

        
        for iTarget in range(CRD0.NTarget):
            fig = pylab.figure("DynSpecMS",
                               #constrained_layout=True,
                               figsize=(15,8))
            gs = fig.add_gridspec(4, 3, wspace=0.05, hspace=0.05)
            fig.clf()
            
            for ipol,pol in enumerate(Lpol):
                ax = fig.add_subplot(gs[ipol,:])
                
                Plot(L_CRD[ipol],ax,iTarget)
                if pol!=Lpol[-1]:
                    frame1=pylab.gca()
                    #frame1.axes.get_xaxis().set_visible(False)
                    #frame1.axes.get_xaxis().set_ticks([])
                    frame1.axes.xaxis.set_ticklabels([])

                
            CRD=L_CRD[ipol]
            pylab.suptitle(CRD.DicoDyn[iTarget]["CD"].Name)
            pylab.draw()
            pylab.show(block=False)
            pylab.pause(0.5)
            
            #FitsName=CRD.DicoDyn[iTarget]["CD"].File.split("/")[-1]
            FitsName=CRD.DicoDyn[iTarget]["CD"].Name
            os.system("mkdir -p %s"%CRD.SaveDir)
            FName="%s/%s.DynSpec.png"%(CRD.SaveDir,FitsName)
            print("Saving fig: %s"%FName)
            fig.savefig(FName)

            DicoLightCurves={}
            
            t0,t1,f0,f1=CRD.DicoDyn[iTarget]["CD"].extent
            NTimesMovie=300#300
            LTimes=(np.linspace(t0,t1,NTimesMovie)).tolist()
            dt=2.5/60
            LTimes=(np.arange(t0,t1,dt)).tolist()
            
            #LTimes=[1.2]
            
            for iTime,Time in enumerate(LTimes):
                print("%i / %i [t=%f]"%(iTime,len(LTimes),Time))
                fig.clf()
                widths = [3, 1, 1]
                gs = fig.add_gridspec(4, 3, wspace=0.13, hspace=0.05, left=0.05, right=0.95, width_ratios=widths)
                
                AllFlagged=False
                for ipol,pol in enumerate(Lpol):
                    # PlotDynSpec
                    ax = fig.add_subplot(gs[ipol,0])
                    Plot(L_CRD[ipol],ax,iTarget,ColorBar=False)
                    ax.plot([Time,Time],[f0,f1],ls="--",color="green",lw=3)

                    
                    if pol!=Lpol[-1]:
                        frame1=pylab.gca()
                        frame1.axes.xaxis.set_ticklabels([])
                    # Plot Slice
                    
                    # x,y=L_CRD[ipol].GiveSlice(iTarget,Time)
                    x,y,ey=L_CRD[ipol].GiveTimeSliceSmooth(iTarget,Time)
                    if np.all(y==0):
                        AllFlagged=True
                        break
                    ex=None

                    
                    # Dt=0.1
                    # nFreq=10
                    # x,ex,y,ey=L_CRD[ipol].GiveTimeSlice(iTarget,Time,Dt,nFreq)
                    DicoLightCurves[pol]={"Type":"TimeSlice",
                                          "xy":(x,y),"exy":(ex,ey)}
                    #ind=np.where(y!=0)[0]

                    ind=(y==0)
                    #x[ind]=np.nan
                    yp=y.copy()
                    eyp=ey.copy()
                    yp[ind]=np.nan
                    eyp[ind]=np.nan


                    ax0 = fig.add_subplot(gs[ipol,1])
                    ax0.plot(x,yp,label=pol,color="black")
                    #ax0.legend()
                    y0=y.copy()
                    y0.fill(0)
                    ax0.plot(x,y0,ls="--",color="black")
                    ax0.fill_between(x,yp+3*eyp,yp-3*eyp,color="gray",alpha=0.3)
                    ax0.fill_between(x,yp+2*eyp,yp-2*eyp,color="gray",alpha=0.3)
                    ax0.fill_between(x,yp+eyp,yp-eyp,color="gray",alpha=0.3)
                    ax0.set_xlabel("Frequency [MHz]")
                    #ax0.set_xlabel("Flux density [mJy]")
                    ax0.set_ylabel("Flux density [mJy]")
                    ax0.yaxis.tick_right()
                    #ax0.yaxis.set_label_position("right")

                    # ax0 = fig.add_subplot(gs[ipol,1])
                    # ax0.plot(y,x,label=pol,color="black")
                    # #ax0.legend()
                    # y0=y.copy()
                    # y0.fill(0)
                    # ax0.plot(y0,x,ls="--",color="black")
                    # ax0.fill_betweenx(x,y+3*ey,y-3*ey,color="gray",alpha=0.3)
                    # ax0.fill_betweenx(x,y+2*ey,y-2*ey,color="gray",alpha=0.3)
                    # ax0.fill_betweenx(x,y+ey,y-ey,color="gray",alpha=0.3)
                    # #ax0.set_xlabel("Frequency [MHz]")
                    # ax0.set_xlabel("Flux density [mJy]")
                    
                    # divider = make_axes_locatable(pylab.gca())
                    # ax0 = divider.append_axes("right", size=1.2, pad=0.1)#, sharey=ax)
                    # ax0.plot(y,x,label=pol,color="black")
                    # ax0.legend()
                    # y0=y.copy()
                    # y0.fill(0)
                    # #ax0.plot(x,y0,ls="--",color="black")
                    # ax0.fill_betweenx(x,y+3*ey,y-3*ey,color="gray",alpha=0.3)
                    # ax0.set_xlabel("Frequency [MHz]")

                    
                
                if AllFlagged:
                    continue
                # ax0.legend()

                x,yQ=DicoLightCurves["Q"]["xy"]
                _,yI=DicoLightCurves["I"]["xy"]
                x,yU=DicoLightCurves["U"]["xy"]
                _,yV=DicoLightCurves["V"]["xy"]
                _,eyI=DicoLightCurves["I"]["exy"]
                _,eyQ=DicoLightCurves["Q"]["exy"]
                _,eyU=DicoLightCurves["U"]["exy"]
                _,eyV=DicoLightCurves["V"]["exy"]
                def giveTheta(Q,U):
                    iQ=Q.copy()
                    iQ[Q==0]=1.
                    iQ=1./iQ
                    #theta=0.5*np.arctan(U*iQ)
                    #theta1=0.5*np.angle(Q+1j*U)
                    theta=0.5*np.angle(Q+1j*U)
                    return theta#,theta1


                theta=giveTheta(yQ,yU)
                theta[yQ==0]=np.nan
                nRand=100
                l=((3e8/(x*1e6))**2)
                L_theta=[]
                ax1=fig.add_subplot(gs[0,2])
                ax2=fig.add_subplot(gs[1,2])
                ax3=fig.add_subplot(gs[2,2],sharex=ax2)
                ax4=fig.add_subplot(gs[3,2],sharex=ax2)

                for iRand in range(nRand):
                    q,u=yQ+np.random.randn(yQ.size)*eyQ,yU+np.random.randn(yU.size)*eyU
                    I,V=yI+np.random.randn(yI.size)*eyI,yV+np.random.randn(yV.size)*eyV

                    M=(eyI==0)
                    eyI0=eyI.copy()
                    eyI0[M]=1
                    SNR=np.abs(I/eyI0)
                    SNR[M]=0
                    SNRMax=5.
                    SNRMin=2.
                    #SNR-=SNRMin
                    SNR[SNR>SNRMax]=SNRMax
                    SNR[SNR<SNRMin]=0
                    SNR/=SNRMax
                    
                    theta_r=giveTheta(q,u)
                    theta_r[theta_r==0]=np.nan
                    P=np.sqrt(q**2+u**2)
                    # c=
                    ax1.scatter(l,theta_r,c="black",alpha=0.01,s=2,linewidth=0)
                    # c=np.zeros((l.size,4),np.float32)
                    # c[:,-1]=SNR*0.05
                    # #print(SNR,SNR.max())
                    # ax1.scatter(l,theta_r,c=c,s=2,linewidth=0)
                    # #ax1.scatter(l,theta_r1,c="gray",alpha=0.05,s=2,linewidth=0)

                    M=(I==0)
                    I[M]=1
                    R=P/I
                    Rv=V/I
                    S=(I-np.sqrt(q**2+u**2+V**2))/I
                    R[M]=np.nan
                    Rv[M]=np.nan
                    S[M]=np.nan

                    ax2.scatter(x,R,c="black",alpha=0.01,s=2,linewidth=0)
                    ax3.scatter(x,Rv,c="black",alpha=0.01,s=2,linewidth=0)
                    ax4.scatter(x,S,c="black",alpha=0.01,s=2,linewidth=0)

                    # ax2.scatter(x,R,c=c,s=2,linewidth=0)
                    # ax3.scatter(x,Rv,c=c,s=2,linewidth=0)
                    # ax4.scatter(x,S,c=c,s=2,linewidth=0)

                    
                    
                    # cmap="seismic"
                    # #cmap="coolwarm"
                    # cmap = matplotlib.cm.get_cmap(cmap).copy()
                    # #cmap.set_bad(color='black')
                    #ax1.hexbin(l,theta_r, gridsize=50,mincnt=2)#, bins="log")#,cmap=cmap, vmin, vmax)
                    
                    # T=x[1]-x[0]
                    # y0=q+u*1j
                    # y1=y0.copy()
                    # y1[y0!=0]=1
                    
                    # N=y0.size


                    
                    # yf0 = p2fd(y0)
                    # yf0/=N
                    
                    # N=y0.size
                    # yf1 = p2fd(y1)
                    # yf1/=N
                    
                    # xf = fftfreq(N, T)#[:N//2]
                    
                    # import RLRMsynth
                    # import math
                    # from math import pi
                    # c = 2.9979e8
                    # yf0=yf1

                    

                    
                    # fclean, residual = RLRMsynth.rmclean(yf0, yf1, xf,plot_every=None)
                    # numax=x.min()*1e6
                    # numin=x.max()*1e6
                    # kmin = (c / numax) ** 2 / pi                            # lambda_min ^ 2 / pi
                    # kmax = (c / numin) ** 2 / pi                            # lambda_max ^ 2 / pi
                    # k0sq = (kmin + kmax)/2
                    # fwhm = 3.8/pi/(kmax-kmin)
                    # rmbin = 10.
                    # sigsq = (fwhm/rmbin)**2/(4.0*math.log(2.0))
                    # fconv, gauss  = RLRMsynth.convolve(fclean, sigsq)

                    # ind=np.argsort(xf)
                    # #ax2.scatter(xf,fconv,color="gray",alpha=0.05,s=5,linewidth=0)
                    # ax2.plot(xf[ind],np.abs(fconv[ind]),color="gray",alpha=0.05)#,s=5,linewidth=0)
                    
                    # op=lambda x: np.abs(x)
                    # ayf0=op(yf0)
                    # ayf1=op(yf1)
                    # Max0=np.max(ayf0)
                    # Max1=np.max(ayf1)
                    # #ayf0-=ayf1*Max0/Max1
                    
                    # #ax2.scatter(xf,ayf0,c="gray",alpha=0.05,s=3,linewidth=0)
                    # ax2.scatter(xf,ayf0,c="gray",alpha=0.05,s=3,linewidth=0)
                    # ax3.scatter(xf,ayf1,c="gray",alpha=0.05,s=3,linewidth=0)
                    
                    
                    #ax2.plot(xf,np.abs(yf))#,c="gray",alpha=0.05,s=1,linewidth=0)

                ax1.grid(color='black', linestyle='--', linewidth=1.5,alpha=0.5)
                ax1.yaxis.tick_right()
                ax1.yaxis.set_label_position("right")
                ax1.xaxis.tick_top()
                ax1.xaxis.set_label_position("top")
                ax1.set_xlabel("$\lambda^2$ [m$^2$]")
                ax1.set_ylabel("Position angle [rad]")
                
                ax2.grid(color='black', linestyle='--', linewidth=1.5,alpha=0.5)
                ax2.set_ylim(-2,2)
                ax2.yaxis.tick_right()
                ax2.yaxis.set_label_position("right")
                ax2.set_ylabel("${\sqrt{Q^2+U^2}} / I$")
                
                ax3.grid(color='black', linestyle='--', linewidth=1.5,alpha=0.5)
                ax3.set_ylim(-2,2)
                ax3.yaxis.tick_right()
                ax3.yaxis.set_label_position("right")
                ax3.set_ylabel("V/I")
                
                ax4.grid(color='black', linestyle='--', linewidth=1.5,alpha=0.5)
                ax4.set_ylim(-2,2)
                ax4.yaxis.tick_right()
                ax4.yaxis.set_label_position("right")
                ax4.set_ylabel("(I-$\sqrt{Q^2+U^2+V^2}$)/I")
                ax4.set_xlabel("Frequency [MHz]")
                
                #ax3.grid(color='black', linestyle='--', linewidth=1.5,alpha=0.5)
                #ax2.set_xlim(-0.03,0.03)
                # ax1.plot(l,theta,color="black")
                # ax1.plot(x,theta,color="black")
                    
                # ax2.set_xlabel("$\phi$ [rad.m$^2]$")
                # ax2.set_ylabel("F")
                # ax2.yaxis.set_label_position("right")
                # ax2.yaxis.tick_right()

                
                pylab.suptitle("Name: %s [ %3.3f h since start]"%(CRD.DicoDyn[iTarget]["CD"].Name,Time))
                #pylab.tight_layout()
                pylab.draw()
                pylab.show(block=False)
                pylab.pause(0.5)
                
                FName="%s/%s.FreqCurve.t_%3.3i.png"%(CRD.SaveDir,FitsName,iTime)
                print("Saving fig: %s"%FName)
                fig.savefig(FName)

        # CRD=ClassRunDir(BaseDir=BaseDir,DB=DB,pol="I",SaveDir=SaveDir)
        # L0=CRD.runDir()
        
        # CRD=ClassRunDir(BaseDir=BaseDir,DB=DB,pol="V",SaveDir=SaveDir)
        # L0=CRD.runDir()
        
        # # CRD=ClassRunDir(BaseDir=BaseDir,DB=DB,pol=1)
        # # L0=CRD.runDir()
        # # CRD=ClassRunDir(BaseDir=BaseDir,DB=DB,pol=2)
        # # L0=CRD.runDir()

        # for it in range(len(L3)):
        #     t=L3[it]
        #     F=t["File"]
        #     tDB=copy.deepcopy(DB[F])
            
        #     tDB["R3"]=t["R"]
            
        #     # t=L0[it]
        #     # F=t["File"]
        #     # tDB["R0"]=t["R"]
            
        #     DBOut.append(tDB)
    Save("D.pickle",DBOut)

def Plot(CRD,ax,iTarget,ColorBar=True):
    CumulTarget=CRD.CumulTarget
    CumulOff=CRD.CumulOff
    DicoDyn=CRD.DicoDyn
    
    ListOut=[]
    # mCumulOff=np.mean(CumulOff,axis=0)
    # ind=np.where((mCumulOff>0.2)&(mCumulOff<0.8))[0]
    # #Std=np.sqrt(np.sum(((CumulOff-mCumulOff)[ind])**2))/ind.size
    # d=(CumulOff-mCumulOff.reshape((1,-1)))
    # Std=np.std(d[:,ind])
    
    Offs=np.array([DicoDyn[i]["CD"].fI for i in range(len(DicoDyn)) if DicoDyn[i]["Type"]=="Off"])
    MeanOff=0#np.mean(Offs,axis=0)
    #Offs=np.array([DicoDyn[i]["CD"].fI for i in range(len(DicoDyn)) if DicoDyn[i]["Type"]=="Off"])
    StdOff=np.std(Offs,axis=0)
    StdOff[StdOff==0]=1

    #ScalarRMS=scipy.stats.median_absolute_deviation(Offs[Offs!=0],axis=None)
    ScalarRMS=scipy.stats.median_abs_deviation(Offs[Offs!=0],axis=None)
    
    i=iTarget
    if DicoDyn[i]["Type"]=="Off": return
    #R=np.sum((CumulTarget[i]-mCumulOff)**2)/CRD.CumulTarget.shape[1]/Std
    FileName=DicoDyn[i]["CD"].File.split("/")[-1]
    R=0
    ListOut.append({"File":FileName,
                    "R":R})
        
    #print(DicoDyn[i]["CD"].File.split("/")[-1],R)
    #if R<0.1 or DicoDyn[i]["CD"].FracFlag>0.4: continue
    
    current = 0#multiprocessing.current_process()._identity[0]
    I=DicoDyn[i]["CD"].fI.copy()
    I[DicoDyn[i]["CD"].Mask]=np.nan
    MeanCorr=False
    if CRD.pol=="L": MeanCorr=True
    cmap="seismic"
    #cmap="coolwarm"

    cmap = matplotlib.cm.get_cmap(cmap).copy()
    cmap.set_bad(color='black')

    
    im=imShow(I,MeanCorr=MeanCorr,extent=DicoDyn[i]["CD"].extent,cmap=cmap,vmin=-10*ScalarRMS,vmax=10*ScalarRMS)#"gray")
    
    x0,x1,y0,y1=DicoDyn[i]["CD"].extent
    nx,ny=I.shape
    X,Y=np.mgrid[x0:x1:1j*nx,y0:y1:1j*ny]

    fIn=(DicoDyn[i]["CD"].fI.copy())/StdOff
    fIn[DicoDyn[i]["CD"].Mask]=np.nan
    Z=fIn
    levels=np.array([-5,-3,3,5]).tolist()#*ScalarRMS
    #imShow(fIn,MeanCorr=MeanCorr,extent=DicoDyn[i]["CD"].extent,cmap=cmap,vmin=-5,vmax=5)#"gray")
    
    #CS=pylab.contour(X, Y, Z, levels,color="black",extent=DicoDyn[i]["CD"].extent)
    CS=pylab.contour(Z, levels,colors="k",extent=DicoDyn[i]["CD"].extent, linewidths=0.5)
    
    #pylab.clabel(CS, inline=1, fontsize=10)
    pylab.grid(color='black', linestyle='--', linewidth=1.5,alpha=0.5)

    pylab.ylabel("Frequency [MHz]")
    pylab.xlabel("Time [hours since %s]"%(DicoDyn[i]["CD"].StrT0.replace("T"," @ ")))
    #pylab.legend(CRD.pol,loc=1)#"upper left")
    ax=pylab.gca()
    props = dict(boxstyle='round',
                 facecolor='white',
                 alpha=0.8)
    ax.text(0.03, 0.95, CRD.pol, transform=ax.transAxes,
            fontsize=17,weight='bold',
            verticalalignment='top', bbox=props)
    if ColorBar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        clb = pylab.colorbar(im, cax=cax)
        #clb.set_label('Flux density [mJy]', labelpad=-40, y=1.05, rotation=0)
        clb.set_label(label='Flux density [mJy]')#,weight='bold')
    
    #pylab.text(0.1,0.9,CRD.pol, fontsize=12)


if __name__=="__main__":
    runAllDir(Patern=sys.argv[1])
