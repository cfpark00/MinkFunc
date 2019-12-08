import numpy as np
import time
import scipy.spatial as sp
import scipy
import matplotlib.path as mpltpath
import matplotlib.pyplot as plt
from scipy import interpolate
rtimewprMFs=[]
rtimesum=[]


print("lsspy.optMF3D: This optimized library computes the 3-D Minkowski Functional based on a Voronoi Tessellation.\n")

C_chi=(4./3)*np.pi #parameter often used

def Voronoi(points):

	return sp.Voronoi(points)

def dist(x1,x2):
    xsep=x1-x2
    return np.sqrt(np.einsum('i,i', xsep, xsep))

def zfastsolang(dtp,rop,one,nprop,npone,sin,cos):
    dtpsq=dtp**2
    return 2*np.arctan((rop*one*sin)/(nprop*npone+dtp*(nprop+npone)+rop*one*cos+dtpsq))

def get_G_ijk(r,h,d12,d23,d31):
    s=(d12+d23+d31)/2
    a=2*np.arctan((2*h*np.sqrt(s*(s-d12)*(s-d23)*(s-d31))/(4*r**2-(d12**2+d23**2+d31**2)/2))/r)
    if a<0:
        return 2*np.pi+a
    return a

def get_G_ijko(r,d12,d23,d31):
    r2=2*r
    a3=2*np.arcsin(d12/r2)
    a1=2*np.arcsin(d23/r2)
    a2=2*np.arcsin(d31/r2)
    return 4*np.arctan(np.sqrt(np.tan((a1+a2+a3)/4)*np.tan((a1+a2-a3)/4)*np.tan((a1-a2+a3)/4)*np.tan((-a1+a2+a3)/4)))
def norm(v):

    return np.sqrt(np.einsum('i,i', v, v))

def normvec(v):

    return v/np.sqrt(np.einsum('i,i', v, v))

def polarea2d(verts):
    xys=verts.T
    return np.abs(np.dot(xys[0][:],np.roll(xys[1][:],-1))-np.dot(xys[1][:],np.roll(xys[0][:],-1)))/2

def verttoedges(verts):
	edges=[]
	for i in range(len(verts)):
		edges.append(np.array([verts[i-1],verts[i]]))
	return np.array(edges)

def kdt(points):

	return sp.cKDTree(points)

def get_norms(rs,size):
    nvols=(size-4*rs)**3
    nareas=(size-4*rs)**2
    nlens=(size-4*rs)
    return [nvols,nareas,nlens]

def getplane_abcd(edges):
    v1=edges[1]-edges[0]
    v2=edges[2]-edges[0]
    n=np.cross(v1,v2)
    return n,-np.dot(n,edges[0])

def getline_abc(edge):

    return np.array([edge[1][1]-edge[0][1],edge[0][0]-edge[1][0]]),edge[0][1]*(edge[1][0]-edge[0][0])-edge[0][0]*(edge[1][1]-edge[0][1])

def disttoplane(x,n,d):
    normsq=np.sum(np.square(n))
    dvun=n.dot(x)+d
    return np.abs(dvun)/np.sqrt(normsq), x-n*dvun/normsq

def center(verts):

    return np.mean(verts,axis=0)

def convcoor(nn,h,verts):
    cent=center(verts)
    xax=normvec(cent-h)
    yax=np.cross(nn,xax)
    vertsp=verts-h
    ys=np.dot(vertsp,yax)
    xs=np.dot(vertsp,xax)
    vertss=np.array([xs,ys]).T
    path = mpltpath.Path(vertss)
    if path.contains_point([0,0]):
        return vertss,True,np.arctan2(ys,xs)
    return vertss,False,np.arctan2(ys,xs)

def vertstoedgesorted(verts):
    # maps the same edges to the exact same objects
    edges=[]
    for i in range(len(verts)):
        if verts[i-1]<=verts[i]:
            edges.append((verts[i-1],verts[i]))
        else:
            edges.append((verts[i],verts[i-1]))
    return edges

class VCells:
    def __init__(self):
        self.pointstowall={}
        self.adjm={}
        self.edgecounter={}
        self.walls={}
        self.wallparams={}
    
    def add_wall(self,idd,ptinds,wall,d,tpoints):
        offflag=False
        if any([x==-1 for x in wall]):
            offflag=True
        if ptinds[0]<=ptinds[1]:
            ind=(ptinds[0],ptinds[1])
        else:
            ind=(ptinds[1],ptinds[0])
        if not offflag:
            self.pointstowall[ind]=idd
            self.walls[idd]=wall
            self.wallparams[idd]=[d,tpoints]
        for edgeinds in vertstoedgesorted(wall):
            if not edgeinds in self.edgecounter:
                self.edgecounter[edgeinds]=1
            else:
                self.edgecounter[edgeinds]+=1
            self.adjm.setdefault(edgeinds,[]).append(d)

    def gen_wall_lists(self,size):
        try:
            self.walllists
            print("Walllists Already Generated")
            return
        except:
            self.walllists=[[] for _ in range(size)]
            print("Generating Walllists")
            st=time.time()
            for pinds,wallidd in self.pointstowall.items():
                self.walllists[pinds[0]].append(wallidd)
                self.walllists[pinds[1]].append(wallidd)
            print("Walllists generated")      

    def gen_wall_params(self,vor,contributingwalls=None):
        print("Wall Based pre-computation running.")
        st=time.time()
        verbosecount=0
        waltotnum=len(self.walls)
        for idd,wall in self.walls.items():
            if verbosecount%200==0 and verbosecount!=0:
                print("\r "+str(100*(verbosecount+1)/waltotnum)+" % Processed",end="")
            verbosecount+=1

            if not contributingwalls==None:
                if idd not in contributingwalls:
                    continue

            x=self.wallparams[idd][1][0]
            y=self.wallparams[idd][1][1]
            verts=vor.vertices[wall]

            maxrforwall=0
            vertdists=[]
            for vert in verts:
                dt=dist(x,vert)
                vertdists.append(dt)
                if dt>maxrforwall:
                    maxrforwall=dt

            n=x-y
            d=-n.dot(x+y)/2
            dtp,h=disttoplane(x,n,d)

            verts2d,inside,thets=convcoor(normvec(n),h,verts)
            wallarea=polarea2d(verts2d)

            vert2dnorms=[]
            for vert2d in verts2d:
                vert2dnorms.append(norm(vert2d))

            solangs=[]
            dtes=[]
            thetas=[]
            sss=[]
            for i,(edge2d,thet2) in enumerate(zip(verttoedges(verts2d),verttoedges(thets))):
                nop,c=getline_abc(edge2d)
                nopnormsq=np.einsum('i,i', nop, nop)
                nopnorm=np.sqrt(nopnormsq)
                dte=np.abs(c)/nopnorm
                dtes.append(dte)
                theta=thet2[1]-thet2[0]
                if inside:#detect crossing at pi
                    if theta<-np.pi:
                        theta+=2*np.pi
                    elif theta>np.pi:
                        theta-=2*np.pi


                s=np.abs(np.sin(theta))
                ccos=np.cos(theta)
                thetas.append((theta,s,ccos))

                xdet=-nop[0]*c/nopnormsq
                if (xdet>edge2d[0][0] and xdet<edge2d[1][0]) or (xdet<edge2d[0][0] and xdet>edge2d[1][0]):
                    sss.append(False)
                else:
                    sss.append(True)

                if theta>=0:
                    solangs.append(zfastsolang(dtp,vert2dnorms[i-1],vert2dnorms[i],vertdists[i-1],vertdists[i],s,ccos))
                else:
                    solangs.append(-zfastsolang(dtp,vert2dnorms[i-1],vert2dnorms[i],vertdists[i-1],vertdists[i],s,ccos))

            fullsolang=np.abs(np.sum(solangs))


            self.wallparams[idd].extend((fullsolang,maxrforwall,wallarea,vertdists,dtp,solangs,dtes,thetas,vert2dnorms,sss, (verts2d,inside)))
        print()
        print("Wall Based precomputation completed. Time: "+str(time.time()-st))
        print()

    def gen_vol(self,rs,size):
        nvols=(size-4*rs)**3
        nareas=(size-4*rs)**2
        nlens=(size-4*rs)
        self.norms=[nvols,nareas,nlens]

    def gen_ptrcellvol(self,size,contributingwalls):
        self.ptrcellvol=[0 for _ in range(size)]
        print("Generating Particle cell volume")
        st=time.time()
        for pinds,idd in self.pointstowall.items():
            if idd not in contributingwalls:
                continue
            thisvol=self.wallparams[idd][4]*self.wallparams[idd][6]/3.
            self.ptrcellvol[pinds[0]]+=thisvol
            self.ptrcellvol[pinds[1]]+=thisvol
        print("Particle cell volume Generated. Time: "+str(time.time()-st))

    def gen_wprMFs(self,rs,contributingwalls=None,accelerate=False,accelerator=None):#one-side computation
        global rtimewprMFs

        print("Calculating Reduced Minkowski Functionals.")
        self.wprMFs=[{} for _ in range(len(rs))]
        n=len(self.walls)
        if accelerate:
            print("Accelerating. This loses physical meaning for the wall partial MFs but they will not affect the pMF of balls.")
            if accelerator==None:
                assert False,"No Accelerator given"
        st=time.time()
        for i,r in enumerate(rs):
            freestoadd=0
            verbosecount=0
            stt=time.time()
            for idd,wall in self.walls.items():
                if not contributingwalls==None:
                    if idd not in contributingwalls:
                        continue

                if verbosecount%200==0 and verbosecount!=0:
                    print("\r Integration "+str(100*(verbosecount+1)/n)+" % Processed",end="")

                if accelerate and (accelerator[0][idd]>r or accelerator[1][idd]<r):
                    continue
                else:
                    dstosend=[]
                    for edgeinds in vertstoedgesorted(wall):
                        if len(self.adjm[edgeinds])!=3:
                            print("NOT 3-cell edge: ",edgeinds,self.adjm[edgeinds],wall)
                        dstosend.append(self.adjm[edgeinds])

                    self.wprMFs[i][idd]=getrMF(idd,r,self.wallparams[idd],dstosend)
            rtime=time.time()-stt
            rtimewprMFs.append(rtime)
            print("\r\tComputation for r= "+str(r)+" done. Time: "+str(rtime),end="")
        print()
        print("Computation Completed. Time: "+str(time.time()-st))
        print()

    def gen_pMF(self,rs,totnum,boundinds,accelerate=False,ptrbound=None):
        global rtimesum
        print("Summing over wall contributions")
        self.pMF=np.full((len(rs),totnum,4),np.nan)
        if accelerate:
            print("Accelerating.")#" Make sure the use of the ptrbound got from gen_wprMFs")
            if ptrbound==None:
                assert False,"No ptrbound or ptrcellvol given."
        st=time.time()
        ttt=0
        for ir,r in enumerate(rs):
            stt=time.time()
            print("\r Summing for r= "+str(r)+" done.",end="")
            contpartind=0
            for i in range(totnum):
                if i in boundinds:
                    self.pMF[ir,i,0]=0
                    self.pMF[ir,i,1]=0
                    self.pMF[ir,i,2]=0
                    self.pMF[ir,i,3]=0
                    continue
                stt=time.time()
                if accelerate and ptrbound[0][i]>=r:
                    self.pMF[ir,i,0]=C_chi*r**3
                    self.pMF[ir,i,1]=C_chi*r**2
                    self.pMF[ir,i,2]=C_chi*r
                    self.pMF[ir,i,3]=C_chi
                    contpartind+=1
                    continue
                if accelerate and ptrbound[1][i]<=r:
                    self.pMF[ir,i,0]=self.ptrcellvol[i]
                    self.pMF[ir,i,1]=0
                    self.pMF[ir,i,2]=0
                    self.pMF[ir,i,3]=0
                    contpartind+=1
                    continue
                M0=0
                M1=0
                M2=0
                M3=0
                for idd in self.walllists[i]:
                    addval=self.wprMFs[ir][idd]
                    M0+=addval[0]
                    M1+=addval[1]
                    M2+=addval[1]/r-addval[2]/2.#Non trivial - sign!! divided by two sheels-> 1/2 factor
                    M3+=addval[1]/r**2-addval[3]/2.+addval[4]/6.#non trivial 1/6 each edge section is accessed 6 times, 2 times per cell
                self.pMF[ir,i,0]=M0
                self.pMF[ir,i,1]=M1
                self.pMF[ir,i,2]=M2
                self.pMF[ir,i,3]=M3
                contpartind+=1
            if contpartind+len(boundinds)!=totnum:
                assert False, "Wrong adding"
            rtime=time.time()-stt
            rtimesum.append(rtime)
        print()
        print("Total pMF generated. Time: "+str(time.time()-st))

    def gen_MF(self,MD0,MD1,MD2,MD3):
        print("Summing over all contributing particles")
        copiedpMF=np.nan_to_num(self.pMF)#to remove nan indicators
        self.MF=np.sum(copiedpMF,axis=1)
        self.MFD=self.MF.copy()
        for i in range(4):#principal kinematical formula
            if i==0:
                self.MFD[:,i]=self.MF[:,i]/MD0
            if i==1:
                self.MFD[:,i]=self.MF[:,i]/MD0#+self.MFD[:,i-1]*MD1/MD0
            if i==2:
                self.MFD[:,i]=self.MF[:,i]/MD0#+(2*self.MFD[:,i-1]*MD1)/MD0#self.MFD[:,i-2]*MD2+
            if i==3:
                self.MFD[:,i]=self.MF[:,i]/MD0#+(3*self.MFD[:,i-2]*MD2+3*self.MFD[:,i-1]*MD1)/MD0#self.MFD[:,i-3]*MD3+
        print("Total MF generated")

    def gen_MF_masked(self,normsize):
        print("Summing over all contributing particles")
        copiedpMF=np.nan_to_num(self.pMF)#to remove nan indicators
        self.MF=np.sum(copiedpMF,axis=1)
        self.MFD=self.MF.copy()
        for i in range(4):#principal kinematical formula
                self.MFD[:,i]=self.MF[:,i]/normsize**3
        print("Total MF generated")

    def checkcount4(self):
        count=0
        for _,val in self.edgecounter.items():
            if val>3:
                count+=1

        return count
    def plotwall(self,idd,r=None):
        fig=plt.figure(figsize=(8,8))
        ax=fig.add_subplot(111,aspect="equal")
        First=0
        for vert in self.wallparams[idd][12][0]:
            if First==0:
                ax.scatter(*vert.T,s=50,c="green")
                First+=1
                continue
            elif First==1:
                ax.scatter(*vert.T,s=25,c="green")
                First+=1
                continue              
            ax.scatter(*vert.T,s=1,c="green")
        for edge in verttoedges(self.wallparams[idd][12][0]):
            ax.plot(*edge.T,lw=1,c="orange")
        ax.scatter([0,0],[0,0],s=3,c="red")

        if not r==None:
            circle1 = plt.Circle([0,0], r, color='red',fill=False)
            ax.add_artist(circle1)


def getrMF(idd,r,wallparam,dss):
    if r<=wallparam[6]:
        return np.array([wallparam[2]*r**3/3,wallparam[2]*r**2/3,0,0,0])

    elif r>=wallparam[3]:
        return np.array([wallparam[4]*wallparam[6]/3.,0,0,0,0])

    else:
        cos=wallparam[6]/r
        alpha=np.pi-2*np.arccos(cos)
        ropsq=r**2-wallparam[6]**2
        rop=np.sqrt(ropsq)
        if wallparam[12][1]:#inside case
            dontevals=[]
            for dte in wallparam[8]:
                if rop>dte:
                    dontevals.append(False)
                else:
                    dontevals.append(True)

            if all(dontevals):
                cc=2*np.pi*rop
                return np.array([wallparam[2]*r**3/3-np.pi*((2*r+wallparam[6])*(r-wallparam[6])**2)/3.,
                    (wallparam[2]-2*np.pi*(1-cos))*r**2/3,
                    (cc*alpha)/6.,
                    wallparam[0]*cc/(3*r*rop),
                    0.])

            
            totsolang=wallparam[2]#we substract from total angle
            totareatovol=0
            totl=0
            G=0
            #area contribution from overlap
            edges=verttoedges(wallparam[12][0])
            packedvertdists=verttoedges(wallparam[10])
            packed3dvertdists=verttoedges(wallparam[5])
            for dte,theta,edge,packedvertdist,dd3,ds,ss,donteval,solang in zip(wallparam[8],wallparam[9],edges,packedvertdists,packed3dvertdists,dss,wallparam[11],dontevals,wallparam[7]):
                thetaabs=np.abs(theta[0])
                if donteval:
                    totareatovol+=thetaabs*ropsq/2.
                    totsolang-=thetaabs*(1-cos)
                    totl+=thetaabs*rop
                elif rop>=packedvertdist[0] and rop>=packedvertdist[1]:
                    totareatovol+=np.abs(np.cross(*edge))/2.
                    totsolang-=np.abs(solang)#zfastsolang(wallparam[6],packedvertdist[0],packedvertdist[1],dd3[0],dd3[1],theta[1],theta[2])

                else:
                    if packedvertdist[0]<=packedvertdist[1]:
                        if rop>=packedvertdist[0]:
                            v21=edge[1]-edge[0]
                            v21n=norm(v21)
                            theta3=np.arcsin(dte/rop)
                            theta2=np.arccos(np.dot(v21,edge[1])/(packedvertdist[1]*v21n))
                            thetaf=theta3-theta2
                            thetan=thetaabs-thetaf
                            sin=np.sin(thetan)
                            totareatovol+=thetaf*ropsq/2.+packedvertdist[0]*rop*np.sin(thetan)/2.
                            totl+=thetaf*rop
                            totsolang-=thetaf*(1-cos)+zfastsolang(wallparam[6],rop,packedvertdist[0],r,dd3[0],sin,np.cos(thetan)) #dtp,rop,one,nprop,npone,sin,cos
                            #print(get_G_ijko(r,*ds)-get_G_ijk(r,np.sqrt(ropsq-dte**2),*ds))
                            G+=get_G_ijko(r,*ds)#get_G_ijk(r,np.sqrt(ropsq-dte**2),*ds)##get_G_ijk(r,np.sqrt(ropsq-dte**2),*ds)
                        elif ss:
                            totareatovol+=thetaabs*ropsq/2.
                            totsolang-=thetaabs*(1-cos)
                            totl+=thetaabs*rop  
                        else:#two crossings
                            intr=np.sqrt(ropsq-dte**2)
                            cccc=dte/rop
                            thetam=2*np.arccos(cccc)
                            thetafn=thetaabs-thetam
                            totareatovol+=intr*dte+thetafn*ropsq/2.

                            totl+=thetafn*rop      
                            totsolang-=thetafn*(1-cos)+zfastsolang(wallparam[6],rop,rop,r,r,np.sin(thetam),2*(cccc)**2-1)#dtp,rop,one,nprop,npone,sin,cos
                            G+=2*get_G_ijko(r,*ds)#get_G_ijk(r,intr,*ds)#get_G_ijko(r,*ds)#get_G_ijk(r,intr,*ds)
                    else:
                        if rop>=packedvertdist[1]:
                            v21=edge[0]-edge[1]
                            v21n=norm(v21)
                            theta3=np.arcsin(dte/rop)
                            theta2=np.arccos(np.dot(v21,edge[0])/(packedvertdist[0]*v21n))
                            thetaf=theta3-theta2
                            thetan=thetaabs-thetaf
                            sin=np.sin(thetan)
                            totareatovol+=thetaf*ropsq/2.+packedvertdist[1]*rop*sin/2.
                            totl+=thetaf*rop
                            totsolang-=thetaf*(1-cos)+zfastsolang(wallparam[6],rop,packedvertdist[1],r,dd3[1],sin,np.cos(thetan)) 
                            G+=get_G_ijko(r,*ds)#get_G_ijk(r,np.sqrt(ropsq-dte**2),*ds)#get_G_ijko(r,*ds)#get_G_ijk(r,np.sqrt(ropsq-dte**2),*ds)

                        elif ss:
                            totareatovol+=thetaabs*ropsq/2.
                            totsolang-=thetaabs*(1-cos)
                            totl+=thetaabs*rop 
                        else:#two crossings
                            intr=np.sqrt(ropsq-dte**2)
                            cccc=dte/rop
                            thetam=2*np.arccos(cccc)
                            thetafn=thetaabs-thetam
                            totareatovol+=intr*dte+thetafn*ropsq/2.
                            totl+=thetafn*rop      
                            totsolang-=thetafn*(1-cos)+zfastsolang(wallparam[6],rop,rop,r,r,np.sin(thetam),2*(cccc)**2-1)
                            G+=2*get_G_ijko(r,*ds)#get_G_ijk(r,intr,*ds)#get_G_ijko(r,*ds)#get_G_ijk(r,intr,*ds)



        else:#outside case
            dontevals=[]
            meeting=False
            edges=verttoedges(wallparam[12][0])
            packedvertdists=verttoedges(wallparam[10])
            polcontrisolang=0#we substract
            totareatovol=0
            totl=0
            G=0
            for packedvertdist,dte,ss in zip(packedvertdists,wallparam[8],wallparam[11]):#Need fix
                if rop>dte:
                    dontevals.append(False)
                    if not meeting:
                        if not ss:
                            meeting=True
                        else:
                            if packedvertdist[0]<rop or packedvertdist[0]<rop:
                                meeting=True
                else:
                    dontevals.append(True)
            if not meeting:
                return np.array([wallparam[2]*r**3/3,wallparam[2]*r**2/3,0,0,0])

            
            packed3dvertdists=verttoedges(wallparam[5])
            for dte,theta,edge,packedvertdist,dd3,ds,ss,donteval,solang in zip(wallparam[8],wallparam[9],edges,packedvertdists,packed3dvertdists,dss,wallparam[11],dontevals,wallparam[7]):
                anticlock=np.sign(theta[0])
                thetaabs=np.abs(theta[0])
                #if i[0]==wallparam[12][3] or i[0]==wallparam[12][2]:
                    #anticlock=-anticlock
                if donteval:
                    totareatovol+=theta[0]*ropsq/2.
                    polcontrisolang+=theta[0]*(1-cos)
                    totl+=theta[0]*rop      
                elif rop>=packedvertdist[0] and rop>=packedvertdist[1]:
                    totareatovol+=anticlock*np.abs(np.cross(*edge))/2.
                    polcontrisolang+=solang#because we already included the sign             

                else:
                    #print("here")
                    if packedvertdist[0]<=packedvertdist[1]:
                        if rop>=packedvertdist[0]:
                            v21=edge[1]-edge[0]
                            v21n=norm(v21)
                            theta3=np.arcsin(dte/rop)
                            theta2=np.arccos(np.dot(v21,edge[1])/(packedvertdist[1]*v21n))
                            thetaf=theta3-theta2
                            thetan=thetaabs-thetaf
                            sin=np.sin(thetan)
                            totareatovol+=anticlock*(thetaf*ropsq/2.+packedvertdist[0]*rop*np.sin(thetan)/2.)
                            totl+=anticlock*thetaf*rop
                            polcontrisolang+=anticlock*(thetaf*(1-cos)+zfastsolang(wallparam[6],rop,packedvertdist[0],r,dd3[0],sin,np.cos(thetan)))      
                            G+=get_G_ijko(r,*ds)#get_G_ijk(r,np.sqrt(ropsq-dte**2),*ds)#get_G_ijko(r,*ds)#get_G_ijk(r,np.sqrt(ropsq-dte**2),*ds)

                        elif ss:
                            totareatovol+=theta[0]*ropsq/2.
                            polcontrisolang+=theta[0]*(1-cos)
                            totl+=theta[0]*rop                             
                        else:#two crossings
                            intr=np.sqrt(ropsq-dte**2)
                            cccc=dte/rop
                            thetam=2*np.arccos(cccc)
                            thetafn=thetaabs-thetam
                            totareatovol+=anticlock*(intr*dte+thetafn*ropsq/2.)
                            totl+=anticlock*thetafn*rop      
                            polcontrisolang+=anticlock*(thetafn*(1-cos)+zfastsolang(wallparam[6],rop,rop,r,r,np.sin(thetam),2*(cccc)**2-1))                   
                            G+=2*get_G_ijko(r,*ds)#get_G_ijk(r,intr,*ds)#get_G_ijko(r,*ds)#get_G_ijk(r,intr,*ds)

                    else:
                        if rop>=packedvertdist[1]:
                            v21=edge[0]-edge[1]
                            v21n=norm(v21)
                            theta3=np.arcsin(dte/rop)
                            theta2=np.arccos(np.dot(v21,edge[0])/(packedvertdist[0]*v21n))
                            thetaf=theta3-theta2
                            thetan=thetaabs-thetaf

                            sin=np.sin(thetan)
                            totareatovol+=anticlock*(thetaf*ropsq/2.+packedvertdist[1]*rop*np.sin(thetan)/2.)
                            totl+=anticlock*thetaf*rop
                            polcontrisolang+=anticlock*(thetaf*(1-cos)+zfastsolang(wallparam[6],rop,packedvertdist[1],r,dd3[1],sin,np.cos(thetan)))
                            G+=get_G_ijko(r,*ds)#get_G_ijk(r,np.sqrt(ropsq-dte**2),*ds)#get_G_ijko(r,*ds)#get_G_ijk(r,np.sqrt(ropsq-dte**2),*ds)

                        elif ss:
                            totareatovol+=theta[0]*ropsq/2.
                            polcontrisolang+=theta[0]*(1-cos)
                            totl+=theta[0]*rop
                        else:#two crossings
                            intr=np.sqrt(ropsq-dte**2)
                            cccc=dte/rop
                            thetam=2*np.arccos(cccc)
                            
                            thetafn=thetaabs-thetam
                            totareatovol+=anticlock*(intr*dte+thetafn*ropsq/2.)
                            totl+=anticlock*thetafn*rop      
                            polcontrisolang+=anticlock*(thetafn*(1-cos)+zfastsolang(wallparam[6],rop,rop,r,r,np.sin(thetam),2*(cccc)**2-1))
                            G+=2*get_G_ijko(r,*ds)#get_G_ijk(r,intr,*ds)#get_G_ijko(r,*ds)#
            #print(totareatovol,polcontrisolang,totl)

            if polcontrisolang>=0:
                totsolang=wallparam[2]-polcontrisolang
            else:
                totsolang=wallparam[2]+polcontrisolang
        totareatovol=np.abs(totareatovol)
        totl=np.abs(totl)
        rM0=totareatovol*wallparam[6]/3.+totsolang*r**3/3
        rM1=totsolang*r**2/3
        rM21=(alpha*totl)/6.
        rM22=wallparam[0]*totl/(3*r*rop)
        rM3=G/3
        return np.array([rM0,rM1,rM21,rM22,rM3])


def MinkFunc3d(rs,points,vor,size,boundary="optimal"):
    global rtimewprMFs,rtimesum
    rtimewprMFs=[]
    rtimesum=[]
    ttttime=time.time()
    totnum=len(points)

    if boundary=="optimal":
        print("Taking optimal boundary")
        boundinds=[]
        contributingwalls={}
        for i,pt in enumerate(points):
            if any([w==-1 for w in vor.regions[vor.point_region[i]]]):
                boundinds.append(i)
                continue
            if any([(v[0]<0 or v[0]>size or v[1]<0 or v[1]>size or v[2]<0 or v[2]>size) for v in vor.vertices[vor.regions[vor.point_region[i]]]]):
                boundinds.append(i)
                continue
        print(str(len(boundinds))+" particles contributing by cell limitations.")

    VC=VCells()
    wallidit=0
    boundedges={}
    boundwalls={}
    for ptinds,wall in vor.ridge_dict.items():
        d=dist(points[ptinds[0]],points[ptinds[1]])
        VC.add_wall(wallidit,ptinds,wall,d,[points[ptinds[0]],points[ptinds[1]]])
        if boundary=="optimal":
            if (ptinds[0] in boundinds):
                if ptinds[1] not in boundinds:
                    boundwalls[wallidit]=True
                    for i,edgeinds in enumerate(vertstoedgesorted(wall)):
                        boundedges.setdefault(edgeinds,[]).extend((np.array(points[ptinds[1]]-points[ptinds[0]]),d,dist(*vor.vertices[[*edgeinds]])))
            if (ptinds[1] in boundinds):
                if ptinds[0] not in boundinds:
                    boundwalls[wallidit]=True
                    for i,edgeinds in enumerate(vertstoedgesorted(wall)):
                        boundedges.setdefault(edgeinds,[]).extend((np.array(points[ptinds[0]]-points[ptinds[1]]),d,dist(*vor.vertices[[*edgeinds]])))
        wallidit+=1
    VC.gen_wall_lists(totnum)

    wls=VC.walllists
    verts=vor.vertices

    if boundary=="mask":
        print("Taking masked boundary")
        boundinds=[]
        contributingwalls={}
        rmax2=2*np.max(rs)
        tr=size-rmax2
        for i,point in enumerate(points):
            if len(wls[i])==0:
                boundinds.append(i)
                continue
            if point[0]<=rmax2 or point[0]>=tr or point[1]<=rmax2 or point[1]>=tr or point[2]<=rmax2 or point[2]>=tr:
                boundinds.append(i)
                continue
            for idd in wls[i]:
                contributingwalls[idd]=True
        normsize=size-2*rmax2
        print(str(len(boundinds))+" particles contributing by cell limitations.")

    if boundary=="optimal":
        contributingwalls={}
        for i,pt in enumerate(points):
            if i in boundinds:
                continue
            for idd in wls[i]:
                contributingwalls[idd]=True
    VC.gen_wall_params(vor,contributingwalls)#contributing for at least one
    VC.gen_ptrcellvol(totnum,contributingwalls)#contributing for at least one
    #print(len(contributingwalls),len(VC.walls))

    if boundary=="optimal":
        print("Generating Window MF.")
        M0tot=0
        M1tot=0
        M2tot=0
        M3tot=4*np.pi/3
        st=time.time()
        for i,pt in enumerate(points):
            if i in boundinds:
                continue
            M0tot+=VC.ptrcellvol[i]

        for idd,_ in boundwalls.items():
            M1tot+=VC.wallparams[idd][4]

        for _,val in boundedges.items():
            if len(val)!=6:
                print("?")
            M2tot+=np.arccos(val[0].dot(val[3])/(val[1]*val[4]))*val[2]
        del boundedges
        M1tot=M1tot/3
        M2tot=M2tot/3
        print("Window MF generated. Time: "+str(time.time()-st))


    rboundsforparticle=[[],[]]
    st=time.time()
    for i,point in enumerate(points):
        if i in boundinds:#boundforalls
            rboundsforparticle[0].append(size)
            rboundsforparticle[1].append(0)
            continue
        rmin=size
        rmax=0
        #print(i)
        for idd in wls[i]:
            thismin=VC.wallparams[idd][0]/2
            #print(idd)
            thismax=VC.wallparams[idd][3]
            if rmin>=thismin:
                rmin=thismin
            if rmax<=thismax:
                rmax=thismax
        rboundsforparticle[0].append(rmin)#only make for valid particles
        rboundsforparticle[1].append(rmax)

    accelerator=[{},{}]
    for ptinds,idd in VC.pointstowall.items():
        #print(ptinds,idd)
        accelerator[0][idd]=min(rboundsforparticle[0][ptinds[0]],rboundsforparticle[0][ptinds[1]])
        accelerator[1][idd]=max(rboundsforparticle[1][ptinds[0]],rboundsforparticle[1][ptinds[1]])

    VC.gen_wprMFs(rs,contributingwalls,accelerate=True,accelerator=accelerator)
    VC.gen_pMF(rs,totnum,boundinds,accelerate=True,ptrbound=rboundsforparticle)
    if boundary=="optimal":
        VC.gen_MF(M0tot,M1tot,M2tot,M3tot)
        tottime=time.time()-ttttime
        print("Grand Sum Time: "+str(tottime)+" in rough, "+str(tottime/((totnum-len(boundinds))*len(rs)))+" per particle per radius.")      
        return VC,boundinds,M0tot,M1tot,M2tot,M3tot,rboundsforparticle
    if boundary=="mask":
        VC.gen_MF_masked(normsize)
        return VC,boundinds,normsize

def getMFTs(rs,rho):#theoretical from poisson
    #numcontpart=n-len(bdi)
    rfts=rs
    V_circ=4*np.pi*rfts**3/3
    A_circ=2*np.pi*rfts**2/3
    L_circ=4*rfts/3
    X_circ=1
    MFTs=[]
    MFTs.append((1-np.exp(-rho*V_circ)))
    MFTs.append(2*rho*A_circ*np.exp(-rho*V_circ))
    MFTs.append(np.pi*rho*(L_circ-3*np.pi*rho*A_circ**2/8)*np.exp(-rho*V_circ))
    MFTs.append(4*np.pi*rho*(X_circ-9*rho*A_circ*L_circ/2+9*np.pi*rho**2*A_circ**3/16)*np.exp(-rho*V_circ)/3)
    return np.array(MFTs)


def interpolatePoissonmSig(rs,rho):
    global sigarr0,sigarr1,sigarr2,sigarr3
    rt=rs/(1/rho)**(1/3)*100 #precalculated at md=100
    return np.array([sigarr0(rt),sigarr1(rt)*(rho/1.0332793625494781e-06)**(1/3),sigarr2(rt)*(rho/1.0332793625494781e-06)**(2/3),sigarr3(rt)*(rho/1.0332793625494781e-06)])#scalings

def plotwithPoissErr(rs,VC,totnum,boundinds,MD0,label="",size=1000):
    rho=(totnum-len(boundinds))/MD0
    MFTs=getMFTs(rs,rho)
    sigs=interpolatePoissonmSig(rs,rho)
    w=0
    fig=plt.figure(figsize=(22,15),dpi=200)
    axs=[]
    for w in range(4):
        axs.append(fig.add_subplot(int("22"+str(w+1))))
        axs[w].scatter(rs,VC.MFD[:,w],s=5,c="black",label=label)
        axs[w].fill_between(rs,MFTs[w,:]-3*sigs[w,:],MFTs[w,:]+3*sigs[w,:],color="red",alpha=0.2,label="3 $\\sigma$ region")
        axs[w].plot(rs,MFTs[w][:],c="blue",lw=1,label="Poisson")
        axs[w].set_xlabel("Evalutation radius r",fontsize=16)
        axs[w].set_ylabel("$w_"+str(w)+"$",fontsize=18)
        axs[w].set_title("N="+str(totnum)+"  L="+str(size),fontsize=18)
        axs[w].legend(fontsize=14)
    axs[2].set_ylim(np.min(MFTs[2][:])*1.1,np.max(MFTs[2][:])*1.1)
    axs[3].set_ylim(np.min(MFTs[3][:])*1.1,np.max(MFTs[3][:])*1.1)
    return fig,axs

def MF3Dplot(rs,VC,totnum,boundinds,normsize):
    neff=totnum-len(boundinds)
    MFTs=getMFTs(rs,neff/normsize**3,neff)
    fig, axs = plt.subplots(4,2)
    fig.set_size_inches(15,5*(4))
    for i in range(4):
        axs[i,0].scatter(rs,VC.MFD[:,i],s=1,label="M-"+str(i))
        axs[i,0].plot(rs,MFTs[i],lw=1,c="blue",label="M-"+str(i))
        axs[i,0].set_xlabel("R")
        axs[i,0].set_ylabel("Functional Values over "+str(normsize)+"^3")
        axs[i,0].legend()
        mar=0.15*(np.max(MFTs[i])-np.min(MFTs[i]))
        axs[i,0].set_ylim(np.min(MFTs[i])-mar,np.max(MFTs[i])+mar)
        axs[i,1].scatter(rs,VC.MFD[:,i],s=1,label="M-"+str(i))
        axs[i,1].set_xlabel("R")
        axs[i,1].set_ylabel("Normalized Functional Values")
        axs[i,1].legend()

def MF3Dplotopt(rs,VC,totnum,boundinds,MD0):
    neff=totnum-len(boundinds)
    MFTs=getMFTs(rs,neff/MD0)#neff/MD0,neff)
    fig, axs = plt.subplots(4,2)
    fig.set_dpi(200)
    fig.set_size_inches(18,5*(4))
    for i in range(4):
        axs[i,0].scatter(rs,VC.MFD[:,i],s=1,label="m"+str(i))
        axs[i,0].plot(rs,MFTs[i],lw=1,c="blue",label="m"+str(i))
        axs[i,0].set_xlabel("r")
        axs[i,0].set_ylabel("Functional Densities")
        axs[i,0].legend()
        mar=0.15*(np.max(MFTs[i])-np.min(MFTs[i]))
        axs[i,0].set_ylim(np.min(MFTs[i])-mar,np.max(MFTs[i])+mar)
        axs[i,1].scatter(rs,VC.MFD[:,i]/(C_chi*rs**(3-i)),s=1,label="M-"+str(i))
        axs[i,1].plot(rs,MFTs[i]/(C_chi*rs**(3-i)),lw=1,c="blue",label="m"+str(i))
        mar=0.15*(np.max(MFTs[i]/(C_chi*rs**(3-i)))-np.min(MFTs[i]/(C_chi*rs**(3-i))))
        axs[i,1].set_ylim(np.min(MFTs[i]/(C_chi*rs**(3-i)))-mar,np.max(MFTs[i]/(C_chi*rs**(3-i)))+mar)
        axs[i,1].set_xlabel("r")
        axs[i,1].set_ylabel("Reduced Functional Density Values")
        axs[i,1].legend()

def plotMF(rs,MFD,totnum,boundinds,MD0):
    neff=totnum-len(boundinds)
    MFTs=getMFTs(rs,neff/MD0)#neff/MD0,neff)
    fig, axs = plt.subplots(4,2)
    fig.set_size_inches(15,5*(4))
    for i in range(4):
        axs[i,0].scatter(rs,MFD[:,i],s=1,label="M-"+str(i))
        axs[i,0].plot(rs,MFTs[i],lw=1,c="blue",label="M-"+str(i))
        axs[i,0].set_xlabel("R")
        axs[i,0].set_ylabel("Functional Values over "+str(MD0))
        axs[i,0].legend()
        mar=0.15*(np.max(MFTs[i])-np.min(MFTs[i]))
        axs[i,0].set_ylim(np.min(MFTs[i])-mar,np.max(MFTs[i])+mar)
        axs[i,1].scatter(rs,MFD[:,i],s=1,label="M-"+str(i))
        axs[i,1].set_xlabel("R")
        axs[i,1].set_ylabel("Normalized Functional Values")
        axs[i,1].legend()

def quietMF3D(rs,points,size):
    global rtimewprMFs,rtimesum
    rtimewprMFs=[]
    rtimesum=[]
    tlog=[]
    ttttime=time.time()
    st=time.time()
    vor=Voronoi(points)
    tlog.append(time.time()-st)
    st=time.time()
    totnum=len(points)
    boundinds=[]
    contributingwalls={}
    for i,pt in enumerate(points):
        if any([w==-1 for w in vor.regions[vor.point_region[i]]]):
            boundinds.append(i)
            continue
        if any([(v[0]<0 or v[0]>size or v[1]<0 or v[1]>size or v[2]<0 or v[2]>size) for v in vor.vertices[vor.regions[vor.point_region[i]]]]):
            boundinds.append(i)
            continue
    VC=VCellsq()
    wallidit=0
    boundedges={}
    boundwalls={}
    for ptinds,wall in vor.ridge_dict.items():
        d=dist(points[ptinds[0]],points[ptinds[1]])
        VC.add_wall(wallidit,ptinds,wall,d,[points[ptinds[0]],points[ptinds[1]]])
        if (ptinds[0] in boundinds):
            if ptinds[1] not in boundinds:
                boundwalls[wallidit]=True
                for i,edgeinds in enumerate(vertstoedgesorted(wall)):
                    boundedges.setdefault(edgeinds,[]).extend((np.array(points[ptinds[1]]-points[ptinds[0]]),d,dist(*vor.vertices[[*edgeinds]])))
        if (ptinds[1] in boundinds):
            if ptinds[0] not in boundinds:
                boundwalls[wallidit]=True
                for i,edgeinds in enumerate(vertstoedgesorted(wall)):
                    boundedges.setdefault(edgeinds,[]).extend((np.array(points[ptinds[0]]-points[ptinds[1]]),d,dist(*vor.vertices[[*edgeinds]])))
        wallidit+=1
    VC.gen_wall_lists(totnum)
    tlog.append(time.time()-st)
    st=time.time()
    wls=VC.walllists
    verts=vor.vertices
    contributingwalls={}
    for i,pt in enumerate(points):
        if i in boundinds:
            continue
        for idd in wls[i]:
            contributingwalls[idd]=True
    VC.gen_wall_params(vor,contributingwalls)#contributing for at least one
    tlog.append(time.time()-st)
    st=time.time()
    VC.gen_ptrcellvol(totnum,contributingwalls)#contributing for at least one
    MD0=0
    st=time.time()
    for i,pt in enumerate(points):
        if i in boundinds:
            continue
        MD0+=VC.ptrcellvol[i]
    tlog.append(time.time()-st)
    st=time.time()
    rboundsforparticle=[[],[]]
    st=time.time()
    for i,point in enumerate(points):
        if i in boundinds:#boundforalls
            rboundsforparticle[0].append(size)
            rboundsforparticle[1].append(0)
            continue
        rmin=size
        rmax=0
        #print(i)
        for idd in wls[i]:
            thismin=VC.wallparams[idd][0]/2
            #print(idd)
            thismax=VC.wallparams[idd][3]
            if rmin>=thismin:
                rmin=thismin
            if rmax<=thismax:
                rmax=thismax
        rboundsforparticle[0].append(rmin)#only make for valid particles
        rboundsforparticle[1].append(rmax)

    accelerator=[{},{}]
    for ptinds,idd in VC.pointstowall.items():
        #print(ptinds,idd)
        accelerator[0][idd]=min(rboundsforparticle[0][ptinds[0]],rboundsforparticle[0][ptinds[1]])
        accelerator[1][idd]=max(rboundsforparticle[1][ptinds[0]],rboundsforparticle[1][ptinds[1]])
    tlog.append(time.time()-st)
    st=time.time()
    VC.gen_wprMFs(rs,contributingwalls,accelerate=True,accelerator=accelerator)
    tlog.append(time.time()-st)
    st=time.time()
    VC.gen_pMF(rs,totnum,boundinds,accelerate=True,ptrbound=rboundsforparticle)
    tlog.append(time.time()-st)
    st=time.time()
    VC.gen_MF(MD0)
    tlog.append(time.time()-st)
    st=time.time()
    tottime=time.time()-ttttime
    tlog.append(tottime)
    print("Grand Sum Time: "+str(tottime)+" in rough, "+str(tottime/((totnum-len(boundinds))*len(rs)))+" per particle per radius.")      
    return VC,boundinds,MD0,[rtimewprMFs,rtimesum],tlog

def MinkowskiFunctional(rs,points,size):
    global rtimewprMFs,rtimesum
    rtimewprMFs=[]
    rtimesum=[]
    tlog=[]
    ttttime=time.time()
    st=time.time()
    vor=Voronoi(points)
    tlog.append(time.time()-st)
    st=time.time()
    totnum=len(points)
    boundinds=[]
    contributingwalls={}
    print("Evaluating boundaries")
    for i,pt in enumerate(points):
        if any([w==-1 for w in vor.regions[vor.point_region[i]]]):
            boundinds.append(i)
            continue
        if any([(v[0]<0 or v[0]>size or v[1]<0 or v[1]>size or v[2]<0 or v[2]>size) for v in vor.vertices[vor.regions[vor.point_region[i]]]]):
            boundinds.append(i)
            continue
    VC=VCellsq()
    wallidit=0
    boundedges={}
    boundwalls={}
    for ptinds,wall in vor.ridge_dict.items():
        d=dist(points[ptinds[0]],points[ptinds[1]])
        VC.add_wall(wallidit,ptinds,wall,d,[points[ptinds[0]],points[ptinds[1]]])
        if (ptinds[0] in boundinds):
            if ptinds[1] not in boundinds:
                boundwalls[wallidit]=True
                for i,edgeinds in enumerate(vertstoedgesorted(wall)):
                    boundedges.setdefault(edgeinds,[]).extend((np.array(points[ptinds[1]]-points[ptinds[0]]),d,dist(*vor.vertices[[*edgeinds]])))
        if (ptinds[1] in boundinds):
            if ptinds[0] not in boundinds:
                boundwalls[wallidit]=True
                for i,edgeinds in enumerate(vertstoedgesorted(wall)):
                    boundedges.setdefault(edgeinds,[]).extend((np.array(points[ptinds[0]]-points[ptinds[1]]),d,dist(*vor.vertices[[*edgeinds]])))
        wallidit+=1
    VC.gen_wall_lists(totnum)
    tlog.append(time.time()-st)
    st=time.time()
    wls=VC.walllists
    verts=vor.vertices
    contributingwalls={}
    for i,pt in enumerate(points):
        if i in boundinds:
            continue
        for idd in wls[i]:
            contributingwalls[idd]=True
    print("Generating Wall Parameters")
    VC.gen_wall_params(vor,contributingwalls)#contributing for at least one
    tlog.append(time.time()-st)
    st=time.time()
    VC.gen_ptrcellvol(totnum,contributingwalls)#contributing for at least one
    MD0=0
    st=time.time()
    for i,pt in enumerate(points):
        if i in boundinds:
            continue
        MD0+=VC.ptrcellvol[i]
    tlog.append(time.time()-st)
    st=time.time()
    rboundsforparticle=[[],[]]
    st=time.time()
    print("Preparing Accelerator")
    for i,point in enumerate(points):
        if i in boundinds:#boundforalls
            rboundsforparticle[0].append(size)
            rboundsforparticle[1].append(0)
            continue
        rmin=size
        rmax=0
        #print(i)
        for idd in wls[i]:
            thismin=VC.wallparams[idd][0]/2
            #print(idd)
            thismax=VC.wallparams[idd][3]
            if rmin>=thismin:
                rmin=thismin
            if rmax<=thismax:
                rmax=thismax
        rboundsforparticle[0].append(rmin)#only make for valid particles
        rboundsforparticle[1].append(rmax)

    accelerator=[{},{}]
    for ptinds,idd in VC.pointstowall.items():
        #print(ptinds,idd)
        accelerator[0][idd]=min(rboundsforparticle[0][ptinds[0]],rboundsforparticle[0][ptinds[1]])
        accelerator[1][idd]=max(rboundsforparticle[1][ptinds[0]],rboundsforparticle[1][ptinds[1]])
    tlog.append(time.time()-st)
    st=time.time()
    print("Wall Functional Calculation")
    VC.gen_wprMFs(rs,contributingwalls,accelerate=True,accelerator=accelerator)
    tlog.append(time.time()-st)
    st=time.time()
    print("Summing over walls")
    VC.gen_pMF(rs,totnum,boundinds,accelerate=True,ptrbound=rboundsforparticle)
    tlog.append(time.time()-st)
    st=time.time()
    VC.gen_MF(MD0)
    tlog.append(time.time()-st)
    st=time.time()
    tottime=time.time()-ttttime
    tlog.append(tottime)
    print("Grand Sum Time: "+str(tottime)+" in rough, "+str(tottime/((totnum-len(boundinds))*len(rs)))+" per particle per radius.")      
    return VC,boundinds,MD0,[rtimewprMFs,rtimesum],tlog

class VCellsq:
    def __init__(self):
        self.pointstowall={}
        self.adjm={}
        self.edgecounter={}
        self.walls={}
        self.wallparams={}
    
    def add_wall(self,idd,ptinds,wall,d,tpoints):
        offflag=False
        if any([x==-1 for x in wall]):
            offflag=True
        if ptinds[0]<=ptinds[1]:
            ind=(ptinds[0],ptinds[1])
        else:
            ind=(ptinds[1],ptinds[0])
        if not offflag:
            self.pointstowall[ind]=idd
            self.walls[idd]=wall
            self.wallparams[idd]=[d,tpoints]
        for edgeinds in vertstoedgesorted(wall):
            if not edgeinds in self.edgecounter:
                self.edgecounter[edgeinds]=1
            else:
                self.edgecounter[edgeinds]+=1
            self.adjm.setdefault(edgeinds,[]).append(d)

    def gen_wall_lists(self,size):
        try:
            self.walllists
            return
        except:
            self.walllists=[[] for _ in range(size)]
            st=time.time()
            for pinds,wallidd in self.pointstowall.items():
                self.walllists[pinds[0]].append(wallidd)
                self.walllists[pinds[1]].append(wallidd)     

    def gen_wall_params(self,vor,contributingwalls=None):
        st=time.time()
        waltotnum=len(self.walls)
        for idd,wall in self.walls.items():

            if not contributingwalls==None:
                if idd not in contributingwalls:
                    continue

            x=self.wallparams[idd][1][0]
            y=self.wallparams[idd][1][1]
            verts=vor.vertices[wall]

            maxrforwall=0
            vertdists=[]
            for vert in verts:
                dt=dist(x,vert)
                vertdists.append(dt)
                if dt>maxrforwall:
                    maxrforwall=dt

            n=x-y
            d=-n.dot(x+y)/2
            dtp,h=disttoplane(x,n,d)

            verts2d,inside,thets=convcoor(normvec(n),h,verts)
            wallarea=polarea2d(verts2d)

            vert2dnorms=[]
            for vert2d in verts2d:
                vert2dnorms.append(norm(vert2d))

            solangs=[]
            dtes=[]
            thetas=[]
            sss=[]
            for i,(edge2d,thet2) in enumerate(zip(verttoedges(verts2d),verttoedges(thets))):
                nop,c=getline_abc(edge2d)
                nopnormsq=np.einsum('i,i', nop, nop)
                nopnorm=np.sqrt(nopnormsq)
                dte=np.abs(c)/nopnorm
                dtes.append(dte)
                theta=thet2[1]-thet2[0]
                if inside:#detect crossing at pi
                    if theta<-np.pi:
                        theta+=2*np.pi
                    elif theta>np.pi:
                        theta-=2*np.pi


                s=np.abs(np.sin(theta))
                ccos=np.cos(theta)
                thetas.append((theta,s,ccos))

                xdet=-nop[0]*c/nopnormsq
                if (xdet>edge2d[0][0] and xdet<edge2d[1][0]) or (xdet<edge2d[0][0] and xdet>edge2d[1][0]):
                    sss.append(False)
                else:
                    sss.append(True)

                if theta>=0:
                    solangs.append(zfastsolang(dtp,vert2dnorms[i-1],vert2dnorms[i],vertdists[i-1],vertdists[i],s,ccos))
                else:
                    solangs.append(-zfastsolang(dtp,vert2dnorms[i-1],vert2dnorms[i],vertdists[i-1],vertdists[i],s,ccos))

            fullsolang=np.abs(np.sum(solangs))


            self.wallparams[idd].extend((fullsolang,maxrforwall,wallarea,vertdists,dtp,solangs,dtes,thetas,vert2dnorms,sss, (verts2d,inside)))


    def gen_ptrcellvol(self,size,contributingwalls):
        self.ptrcellvol=[0 for _ in range(size)]
        st=time.time()
        for pinds,idd in self.pointstowall.items():
            if idd not in contributingwalls:
                continue
            thisvol=self.wallparams[idd][4]*self.wallparams[idd][6]/3.
            self.ptrcellvol[pinds[0]]+=thisvol
            self.ptrcellvol[pinds[1]]+=thisvol

    def gen_wprMFs(self,rs,contributingwalls=None,accelerate=False,accelerator=None):#one-side computation
        global rtimewprMFs
        self.wprMFs=[{} for _ in range(len(rs))]
        n=len(self.walls)
        if accelerate:
            if accelerator==None:
                assert False,"No Accelerator given"
        st=time.time()
        for i,r in enumerate(rs):
            freestoadd=0
            stt=time.time()
            for idd,wall in self.walls.items():
                if not contributingwalls==None:
                    if idd not in contributingwalls:
                        continue

                if accelerate and (accelerator[0][idd]>r or accelerator[1][idd]<r):
                    continue
                else:
                    dstosend=[]
                    for edgeinds in vertstoedgesorted(wall):
                        if len(self.adjm[edgeinds])!=3:
                            print("NOT 3-cell edge: ",edgeinds,self.adjm[edgeinds],wall)
                        dstosend.append(self.adjm[edgeinds])

                    self.wprMFs[i][idd]=getrMF(idd,r,self.wallparams[idd],dstosend)
            rtime=time.time()-stt
            rtimewprMFs.append(rtime)

    def gen_pMF(self,rs,totnum,boundinds,accelerate=False,ptrbound=None):
        global rtimesum
        self.pMF=np.full((len(rs),totnum,4),np.nan)
        if accelerate:
            if ptrbound==None:
                assert False,"No ptrbound or ptrcellvol given."
        st=time.time()
        ttt=0
        for ir,r in enumerate(rs):
            stt=time.time()
            contpartind=0
            for i in range(totnum):
                if i in boundinds:
                    self.pMF[ir,i,0]=0
                    self.pMF[ir,i,1]=0
                    self.pMF[ir,i,2]=0
                    self.pMF[ir,i,3]=0
                    continue
                stt=time.time()
                if accelerate and ptrbound[0][i]>=r:
                    self.pMF[ir,i,0]=C_chi*r**3
                    self.pMF[ir,i,1]=C_chi*r**2
                    self.pMF[ir,i,2]=C_chi*r
                    self.pMF[ir,i,3]=C_chi
                    contpartind+=1
                    continue
                if accelerate and ptrbound[1][i]<=r:
                    self.pMF[ir,i,0]=self.ptrcellvol[i]
                    self.pMF[ir,i,1]=0
                    self.pMF[ir,i,2]=0
                    self.pMF[ir,i,3]=0
                    contpartind+=1
                    continue
                M0=0
                M1=0
                M2=0
                M3=0
                for idd in self.walllists[i]:
                    addval=self.wprMFs[ir][idd]
                    M0+=addval[0]
                    M1+=addval[1]
                    M2+=addval[1]/r-addval[2]/2.#Non trivial - sign!! divided by two sheels-> 1/2 factor
                    M3+=addval[1]/r**2-addval[3]/2.+addval[4]/6.#non trivial 1/6 each edge section is accessed 6 times, 2 times per cell
                self.pMF[ir,i,0]=M0
                self.pMF[ir,i,1]=M1
                self.pMF[ir,i,2]=M2
                self.pMF[ir,i,3]=M3
                contpartind+=1
            if contpartind+len(boundinds)!=totnum:
                assert False, "Wrong adding"
            rtime=time.time()-stt
            rtimesum.append(rtime)

    def gen_MF(self,MD0):
        copiedpMF=np.nan_to_num(self.pMF)#to remove nan indicators
        self.MF=np.sum(copiedpMF,axis=1)
        self.MFD=self.MF.copy()
        for i in range(4):#principal kinematical formula
            if i==0:
                self.MFD[:,i]=self.MF[:,i]/MD0
            if i==1:
                self.MFD[:,i]=self.MF[:,i]/MD0
            if i==2:
                self.MFD[:,i]=self.MF[:,i]/MD0
            if i==3:
                self.MFD[:,i]=self.MF[:,i]/MD0
        print("Total MF generated")

    def gen_MF_masked(self,normsize):
        copiedpMF=np.nan_to_num(self.pMF)#to remove nan indicators
        self.MF=np.sum(copiedpMF,axis=1)
        self.MFD=self.MF.copy()
        for i in range(4):#principal kinematical formula
                self.MFD[:,i]=self.MF[:,i]/normsize**3

    def checkcount4(self):
        count=0
        for _,val in self.edgecounter.items():
            if val>3:
                count+=1

        return count
    def plotwall(self,idd,r=None):
        fig=plt.figure(figsize=(8,8))
        ax=fig.add_subplot(111,aspect="equal")
        First=0
        for vert in self.wallparams[idd][12][0]:
            if First==0:
                ax.scatter(*vert.T,s=50,c="green")
                First+=1
                continue
            elif First==1:
                ax.scatter(*vert.T,s=25,c="green")
                First+=1
                continue              
            ax.scatter(*vert.T,s=1,c="green")
        for edge in verttoedges(self.wallparams[idd][12][0]):
            ax.plot(*edge.T,lw=1,c="orange")
        ax.scatter([0,0],[0,0],s=3,c="red")

        if not r==None:
            circle1 = plt.Circle([0,0], r, color='red',fill=False)
            ax.add_artist(circle1)

PREVALrs=np.array([1.0000000000000002 ,5.0769230769230775 ,9.153846153846155 ,13.230769230769234 ,17.30769230769231 ,21.384615384615387 ,25.461538461538467 ,29.538461538461544 ,33.61538461538462 ,37.6923076923077 ,41.769230769230774 ,45.846153846153854 ,49.923076923076934 ,54.00000000000001 ,58.07692307692309 ,62.15384615384616 ,66.23076923076924 ,70.30769230769232 ,74.3846153846154 ,78.46153846153847 ,82.53846153846155 ,86.61538461538463 ,90.69230769230771 ,94.76923076923079 ,98.84615384615387 ,102.92307692307693 ,107.00000000000001 ,111.0769230769231 ,115.15384615384617 ,119.23076923076925 ,123.30769230769232 ,127.3846153846154 ,131.46153846153848 ,135.53846153846155 ,139.61538461538464 ,143.6923076923077 ,147.7692307692308 ,151.84615384615387 ,155.92307692307693 ,160.00000000000003])
PREVALsig0=np.array([9.30462176053738e-08 ,1.2164382893896889e-05 ,7.11580009464479e-05 ,0.00021366816352618345 ,0.00047339948426560114 ,0.0008784927342966256 ,0.0014492630911988433 ,0.00218903610691815 ,0.0030855998475815155 ,0.004117876422575952 ,0.005250331571820094 ,0.006410840682562735 ,0.007523198493887959 ,0.008516351435331061 ,0.009314483117036997 ,0.00984079546588061 ,0.010064106041359875 ,0.009955977718339285 ,0.009516307085349915 ,0.008784272407092846 ,0.007828170837876977 ,0.0067406857319440155 ,0.005611590573025235 ,0.004519911040650017 ,0.0035241229961765173 ,0.002661674049005517 ,0.0019514477741987815 ,0.0013892386866401335 ,0.0009586740776102773 ,0.0006424760908948394 ,0.0004190694740504398 ,0.0002674488657161619 ,0.00016809828236140587 ,0.00010538962123491593 ,6.723127960685978e-05 ,4.460444384213704e-05 ,3.092263016049216e-05 ,2.201595960806179e-05 ,1.5811746979260698e-05 ,1.1343990423038426e-05])
PREVALsig1=np.array([6.205482265997426e-08 ,1.5964453796324757e-06 ,5.178534591558683e-06 ,1.0721707063632344e-05 ,1.8084354041941446e-05 ,2.7055983864657494e-05 ,3.7245596467963366e-05 ,4.7632405378711977e-05 ,5.805404155886676e-05 ,6.854895537751146e-05 ,7.750427106285083e-05 ,8.424154272426719e-05 ,8.950204693835649e-05 ,9.396238639894009e-05 ,9.754426868844942e-05 ,0.00010147047945357129 ,0.00010519221832255606 ,0.0001111016416550677 ,0.0001175664071604494 ,0.00012295250645925624 ,0.00012462362905966837 ,0.00012120099778592739 ,0.00011310687418981412 ,0.00010123547239255929 ,8.734538311048294e-05 ,7.230024873535402e-05 ,5.730288846404285e-05 ,4.400490801993274e-05 ,3.2798373057837294e-05 ,2.351238500125073e-05 ,1.6219811775974408e-05 ,1.0779697704629692e-05 ,7.071550705881196e-06 ,4.458749748534594e-06 ,2.7497114217189247e-06 ,1.649967683203292e-06 ,1.0301121821781645e-06 ,6.698563309830148e-07 ,4.6583346050687896e-07 ,3.32528592831236e-07])
PREVALsig2=np.array([3.1049734658241755e-08 ,1.578781184586323e-07 ,2.930204442299182e-07 ,4.436663523959398e-07 ,6.350693355299665e-07 ,8.764135617970062e-07 ,1.152498000103475e-06 ,1.4723687712133555e-06 ,1.792300080517532e-06 ,2.0831222543505688e-06 ,2.357229933581647e-06 ,2.5620671392075264e-06 ,2.766214354189731e-06 ,2.8043623209988936e-06 ,2.8803547494717237e-06 ,2.8469831281913246e-06 ,2.8184544131534166e-06 ,2.816877169803228e-06 ,2.6744025653853145e-06 ,2.447635439327259e-06 ,2.1344606672503535e-06 ,1.863428322712351e-06 ,1.6969241289109173e-06 ,1.5763878965798777e-06 ,1.5243877384998714e-06 ,1.398867242324998e-06 ,1.257716178672753e-06 ,1.0404616747478394e-06 ,8.635345595113669e-07 ,6.801082391944476e-07 ,5.163847315459505e-07 ,3.7403457982187703e-07 ,2.5875438814312876e-07 ,1.774648075731325e-07 ,1.196881815491917e-07 ,7.632062375331184e-08 ,4.621024206666922e-08 ,2.7662927906730037e-08 ,1.809344782899971e-08 ,1.2811760195453316e-08])
PREVALsig3=np.array([7.225953663927309e-10 ,8.776091154105139e-09 ,2.0667198888407614e-08 ,3.434099158419224e-08 ,5.3710341601320675e-08 ,6.669689542426629e-08 ,8.495968125143614e-08 ,9.612584899225648e-08 ,1.0498647269978588e-07 ,1.0591133090066566e-07 ,1.1306276448433496e-07 ,1.1075389791870312e-07 ,1.1144771961951519e-07 ,1.1580329598893086e-07 ,1.1411867140064696e-07 ,1.199451065772577e-07 ,1.2416517298034608e-07 ,1.1999724145498257e-07 ,1.0732587318008887e-07 ,1.049280720090045e-07 ,9.55685508380541e-08 ,9.252294272094269e-08 ,7.737807752587672e-08 ,7.050581233003334e-08 ,5.683728323298476e-08 ,4.814726696920971e-08 ,4.086531141604165e-08 ,3.287701261057209e-08 ,2.6783037646764965e-08 ,2.1639817077430443e-08 ,1.6966697888024576e-08 ,1.3402852106480935e-08 ,1.0349915798413502e-08 ,7.026989430110767e-09 ,5.0627635804617635e-09 ,3.737863924645568e-09 ,2.4079666162927574e-09 ,1.5413959406184199e-09 ,8.874062678373296e-10 ,4.5903648646148474e-10])
sigarr0=interpolate.interp1d(PREVALrs,PREVALsig0,bounds_error=False,fill_value="extrapolate")
sigarr1=interpolate.interp1d(PREVALrs,PREVALsig1,bounds_error=False,fill_value="extrapolate")
sigarr2=interpolate.interp1d(PREVALrs,PREVALsig2,bounds_error=False,fill_value="extrapolate")
sigarr3=interpolate.interp1d(PREVALrs,PREVALsig3,bounds_error=False,fill_value="extrapolate")













