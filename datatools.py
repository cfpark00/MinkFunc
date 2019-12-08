import numpy as np
import time
import scipy.spatial as sp
import scipy.signal as sig
import scipy
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
from mpl_toolkits.mplot3d import Axes3D

print("lsspy.datatools: This module handles data in general. It implements a quad-tree and an oct-tree structure so that a hierarchical algorithm can be applied.\n")

overlapped=[]

def read(filename,source="M",mode="xonly",lims=[[0,64],[0,64],[0,64]],maxnum=None,maxline=None):
	xdat=[]
	vdat=[]
	mdat=[]
	if source=="M":
		print("Opening: "+filename)
		file=open(filename,"r")
		partcount=0
		linecount=0
		for line in file:
			if partcount==maxnum:
				break
			if linecount==maxline:
				break
			arr=line.split()
			x=float(arr[0])
			y=float(arr[1])
			z=float(arr[2])
			vx=float(arr[3])
			vy=float(arr[4])
			vz=float(arr[5])
			m=float(arr[-7])
			#print(x,y,z,vx,vy,vz,m)
			if x>lims[0][0] and x<lims[0][1] and y>lims[1][0] and y<lims[1][1] and z>lims[2][0] and z<lims[2][1]:
				xdat.append([x,y,z])
				vdat.append([vx,vy,vz])
				mdat.append(m)
				partcount+=1
				if partcount%1000==0:
					print("\r"+" Paricles read: "+str(partcount)+".", end=" ")
			linecount+=1
		file.close()
		xdat=np.array(xdat)
		vdat=np.array(vdat)
		mdat=np.array(mdat)
		print("")
		print("Total x-data shape: "+str(xdat.shape))
	if source=="B":
		print("Opening: "+filename)
		file=open(filename,"r")
		partcount=0
		linecount=0
		for line in file:
			if line[0]=="#":
				continue
			if partcount==maxnum:
				break
			if linecount==maxline:
				break
			arr=line.split()
			x=float(arr[17])
			y=float(arr[18])
			z=float(arr[19])
			vx=float(arr[20])
			vy=float(arr[21])
			vz=float(arr[22])
			m=float(arr[10])/1e10
			#print(x,y,z,vx,vy,vz,m)
			if x>lims[0][0] and x<lims[0][1] and y>lims[1][0] and y<lims[1][1] and z>lims[2][0] and z<lims[2][1]:
				xdat.append([x,y,z])
				vdat.append([vx,vy,vz])
				mdat.append(m)
				partcount+=1
				if partcount%1000==0:
					print("\r"+" Paricles read: "+str(partcount)+".", end=" ")
			linecount+=1
		file.close()
		xdat=np.array(xdat)
		vdat=np.array(vdat)
		mdat=np.array(mdat)
		print("")
		print("Total x-data shape: "+str(xdat.shape))
	return xdat,vdat,mdat

def octdiv(xdat,sidenum,size,convert=True,comoves=None):
	totnum=len(xdat)
	octs=[]
	comvs=[[] for _ in range(len(comoves))]
	for i in range(sidenum):
		octs.append([])
		for comv in comvs:
			comv.append([])
		for j in range(sidenum):
			octs[i].append([])
			for comv in comvs:
				comv[i].append([])
			for k in range(sidenum):
				octs[i][j].append([])
				for comv in comvs:
					comv[i][j].append([])
	for ii in range(len(xdat)):
		i=int(xdat[ii][0]//size)
		j=int(xdat[ii][1]//size)
		k=int(xdat[ii][2]//size)
		if convert:
			if not comoves==None:
				octs[i][j][k].append(xdat[ii]-np.array([i*size,j*size,k*size]))
				for comv,com in zip(comvs,comoves):
					comv[i][j][k].append(com[ii])
			else:
				octs[i][j][k].append(xdat[ii]-np.array([i*size,j*size,k*size]))
		else:
			if not comoves==None:
				octs[i][j][k].append(xdat[ii])
				for comv,com in zip(comvs,comoves):
					comv[i][j][k].append(com[ii])
			else:
				octs[i][j][k].append(xdat[ii])
	for i in range(sidenum):
		for j in range(sidenum):
			for k in range(sidenum):
				octs[i][j][k]=np.array(octs[i][j][k])
	if comoves==None:
		return np.array(octs)
	return np.array(octs),comvs

def wrandomdownsample(xdat,portion,weights=None):
	newxdat=[]
	maxweight=np.max(weights)
	meanweight=np.mean(weights)
	weights=weights/maxweight
	portion=maxweight*portion/meanweight
	for i in range(len(xdat)):
		if np.random.random()<=portion*weights[i]:
			newxdat.append(xdat[i])
	return np.array(newxdat)

def wrandomdownsamplemultbatch(batches,xdat,portion,weights=None):
	newxdat=[[] for _ in range(batches)]
	meanweight=np.mean(weights)
	weights=portion*weights/meanweight
	for i in range(len(xdat)):
		for j in range(batches):
			if np.random.random()<=weights[i]:
				newxdat[j].append(xdat[i])
	for j in range(batches):
		newxdat[j]=np.array(newxdat[j])
	return np.array(newxdat)

def massweightdownsample(xdat,portion,mdat=None):
	newxdat=[]
	totnum=len(mdat)
	totweight=np.sum(mdat)
	prob=totnum*mdat/totweight
	for i in range(len(xdat)):
		if np.random.random()<=portion*prob[i]:
			newxdat.append(xdat[i])
	return np.array(newxdat)

def downsample(xdat,portion):
	newxdat=[]
	for i in range(len(xdat)):
		if np.random.random()<=portion:
			newxdat.append(xdat[i])
	return np.array(newxdat)

def massthreshold(xdat,mdat,thres):
	newxdat=[]
	newmdat=[]
	for x,m in zip(xdat,mdat):
		if m>=thres:
			newxdat.append(x)
			newmdat.append(m)
	return np.array(newxdat),np.array(newmdat)

def readmult(filename,source="B",n=2,side=250,mode="xonly",maxnum=None,maxline=None):
	xdats=[[] for _ in range(n**3)]
	vdats=[[] for _ in range(n**3)]
	mdats=[[] for _ in range(n**3)]
	if source=="C":
		xind=17
		yind=18
		zind=19
		xmul=1
		vxind=20
		vyind=21
		vzind=22
		mind=36
		mmul=1./1e10
	if source=="M":
		xind=0
		yind=1
		zind=2
		xmul=1
		vxind=3
		vyind=4
		vzind=5
		mind=-7
		mmul=1
	if source=="B":
		xind=17
		yind=18
		zind=19
		xmul=1
		vxind=20
		vyind=21
		vzind=22
		mind=36
		mmul=1./1e10
	if source=="HL":
		xind=2
		yind=3
		zind=4
		xmul=3000./524288
		vxind=5
		vyind=6
		vzind=7
		mind=1
		mmul=2.24e2
	if source=="HT":
		xind=2
		yind=3
		zind=4
		xmul=2000./524288
		vxind=5
		vyind=6
		vzind=7
		mind=1
		mmul=2.22e2
	print("Opening: "+filename)
	file=open(filename,"r")
	partcount=0
	linecount=0
	for line in file:
		if line[0]=="#":
			continue
		if partcount==maxnum:
			break
		if linecount==maxline:
			break
		arr=line.split()
		x=float(arr[xind])*xmul
		y=float(arr[yind])*xmul
		z=float(arr[zind])*xmul
		vx=float(arr[vxind])
		vy=float(arr[vyind])
		vz=float(arr[vzind])
		m=float(arr[mind])*mmul
		loc=int(x//side)+n*int(y//side)+int(z//side)*n**2
		xdats[loc].append([x,y,z])
		vdats[loc].append([vx,vy,vz])
		mdats[loc].append(m)
		partcount+=1
		if partcount%1000==0:
			print("\r"+" Paricles read: "+str(partcount)+".", end=" ")
		linecount+=1
	file.close()
	print("")
	for i in range(n**3):
		xdats[i]=np.array(xdats[i])
		vdats[i]=np.array(vdats[i])
		mdats[i]=np.array(mdats[i])
		print("Total x-data shape for "+str(indtooctforsave(i,n))+": "+str(xdats[i].shape))
	return xdats,vdats,mdats


def autoprocessfile(filename,savedir,source,n,side,z="0"):
	xdats,vdats,mdats=readmult(filename,source=source,n=n,side=side)
	for i in range(n**3):
		xdats[i],vdats[i],mdats[i]=gluedups(xdats[i],vdats[i],mdats[i])
		nnn=indtooctforsave(i,n)
		np.savez(savedir+"/"+source+"-"+z+"-"+str(nnn[0])+str(nnn[1])+str(nnn[2])+".npz",xdat=xdats[i],vdat=vdats[i],mdat=mdats[i])
		print("Saved as: "+source+"-"+z+"-"+str(nnn[0])+str(nnn[1])+str(nnn[2])+".npz")

def indtooctforsave(i,n):
	return (i-n*(i//n)),(i-n**2*(i//n**2))//n,i//(n**2)

def indtoquad(i):
    if i<2:
        if i==0:
            return 0,0
        return 1,0
    if i==2:
        return 0,1
    return 1,1

def indtooct(i):
	if i<4:
		if i<2:
			if i==0:
				return 0,0,0
			return 1,0,0
		if i==2:
			return 0,1,0
		return 1,1,0
	if i<6:
		if i==4:
			return 0,0,1
		return 1,0,1
	if i==6:
		return 0,1,1
	return 1,1,1

def quadtoind(q):
	return q[0]+2*q[1]

def octtoind(q):
	return q[0]+2*q[1]+4*q[2]

def filtersingularities(xdat,halfsize,mdat=None,origin=np.array([0.,0.,0.])):
	t,overlapped,newxdat,newmdat=maketree(xdat,halfsize,mdat,origin=origin)
	del t
	print()
	print("Checking Overlapped")
	checkoverlapped(overlapped,totpartnum=len(xdat),formattedxdatlen=len(newmdat))
	del overlapped
	if mdat==None:
		del newmdat
		return newxdat
	return newxdat,newmdat

def gluedups(xdat,vdat,mdat):
	totnum=len(mdat)
	filteredx,inds=np.unique(xdat,return_inverse=True,axis=0)
	mergednum=totnum-len(filteredx)
	print(str(mergednum)+" duplicates found. Merging:")
	filteredv=np.zeros(filteredx.shape)
	filteredm=np.zeros(filteredx.shape[0])
	for i,ind in enumerate(inds):
		if i%(totnum//100)==0 and i!=0:
			print("\r About "+str(100*(i+1)/totnum)+"% done over "+str(totnum)+" particles.",end="")
		filteredv[ind]+=vdat[i]*mdat[i]
		filteredm[ind]+=mdat[i]
	filteredv=np.divide(filteredv,filteredm[:,None])
	print()
	print("Momentums merged for "+str(mergednum)+" particles")
	return filteredx,filteredv,filteredm


def printlines(file,n=1):
	count=0
	file=open(file,"r")
	for line in file:
		print(str(count)+": "+line)
		print("Splitted: "+str(line.split())+"\n")
		if count==n:
			break
		count+=1

class parttree2d:
	def __init__(self,m,cm,halfsize,coor):
		self.m=m
		self.cm=cm
		self.tree=None
		self.coor=coor
		self.halfsize=halfsize
		self.den=0
	def __str__(self):
		if self.tree==None:
		    return str(self.cm)
		return str("["+str(self.tree[0])+str(self.tree[1])+str(self.tree[2])+str(self.tree[3])+"]")
	def deepen(self):
		self.tree=[None,None,None,None]
		m=self.m
		self.m=0
		cm=self.cm
		self.cm=np.array([0.,0.])
		self.add_particle(m,cm)
	def add_particle(self,m,cm):
		global overlapped
		if self.tree==None:
			if self.halfsize<1e-8:
				print("\r These particles looks overlapped: ",str(cm)," and ",str(self)," will add mass.",end="")
				self.m+=m
				overlapped.append([self.cm,cm])
				return
			self.deepen()
			self.add_particle(m,cm)
			return
		quad=((cm-self.coor)/self.halfsize).astype(int)
		ind=quadtoind(quad)
		if self.tree[ind]==None:
			self.cm=(self.cm*self.m+cm*m)/(self.m+m)
			self.m=self.m+m
			self.tree[ind]=parttree2d(m,cm,self.halfsize/2,self.coor+quad*self.halfsize)
			return
		self.tree[ind].add_particle(m,cm)
	def write_data(self,writearr):
		if self.tree==None:
			writearr[0].append(self.cm)
			writearr[1].append(self.m)
			return
		for t in self.tree:
			if t==None:
				continue
			t.write_data(writearr)

class parttree3d:
	def __init__(self,m,cm,halfsize,coor):
		self.m=m
		self.cm=cm
		self.tree=None
		self.coor=coor
		self.halfsize=halfsize
		self.den=0
	def __str__(self):
		if self.tree==None:
		    return str(self.cm)
		return str("["+str(self.tree[0])+str(self.tree[1])+str(self.tree[2])+str(self.tree[3])+str(self.tree[4])+str(self.tree[5])+str(self.tree[6])+str(self.tree[7])+"]")
	def deepen(self):
		#temp=self.tree
		self.tree=[None,None,None,None,None,None,None,None]
		m=self.m
		self.m=0
		cm=self.cm
		self.cm=np.array([0.,0.,0.])
		self.add_particle(m,cm)
	def add_particle(self,m,cm):
		global overlapped
		if self.tree==None:
			if self.halfsize<1e-8:
				print("\r These particles looks overlapped: ",str(cm)," and ",str(self)," will add mass.",end="")
				self.m+=m
				overlapped.append([self.cm,cm])
				return
			self.deepen()
			self.add_particle(m,cm)
			return
		octt=((cm-self.coor)/self.halfsize).astype(int)
		ind=octtoind(octt)
		if self.tree[ind]==None:
			self.cm=(self.cm*self.m+cm*m)/(self.m+m)
			self.m=self.m+m
			self.tree[ind]=parttree3d(m,cm,self.halfsize/2,self.coor+octt*self.halfsize)
			return
		self.tree[ind].add_particle(m,cm)
	def write_data(self,writearr):
		if self.tree==None:
			writearr[0].append(self.cm)
			writearr[1].append(self.m)
			return
		for t in self.tree:
			if t==None:
				continue
			t.write_data(writearr)
	def getdepth(self):#calculate all depths
		try:
			return self.depth
		except:
			if self.tree==None:
				self.depth=0
				return self.depth
			maxchilddepth=0
			for t in self.tree:
				if t==None:
					continue
				cd=t.getdepth()
				if cd>maxchilddepth:
					maxchilddepth=cd
			self.depth=maxchilddepth+1
			return self.depth
	def N_at_level(self,countsize=None):#big level is small cell
		if self.tree==None:
			return 1
		ncount=0
		if self.halfsize<countsize:
			return 1
		for t in self.tree:
			if t==None:
				continue
			ncount+=t.N_at_level(countsize)
		return ncount
	def N_at_level_saving(self,arr,countsize=None):#big level is small cell
		if self.tree==None:
			return 1
		ncount=0
		if self.halfsize<countsize:
			return 1
		for t in self.tree:
			if t==None:
				continue
			ncount+=t.N_at_level(countsize)
		return ncount


def MBDplot(t,totn,clev=0):
	nss=[]
	for i in range(clev,t.getdepth()):
		nss.append(t.N_at_level(1/2**i))
	x=-np.log(2**np.linspace(clev+1,t.getdepth(),t.getdepth()-clev))
	y=np.log(nss)/np.log(totn)

	ps=scipy.polyfit(x[0:2],y[0:2],1)
	plt.scatter(x,y,s=50,c="black",marker="x")
	plt.plot(x,ps[0]*x+ps[1],ls="--",label="Saturation")
	plt.plot([-15,0],[1,1],ls="--",label="Resolution Limit")
	plt.ylim(0,1.2)
	plt.xlim(-15,0)
	plt.title("Box Counting plot for N="+str(totn)+" from Millenium")
	plt.xlabel("$ln(\epsilon)$")
	plt.ylabel("$ln(N(\epsilon))$")
	plt.legend()
	print(-ps[0])
	return x,y


def fieldplot(field,zrange=[20,30],spanner=np.array,logbiaser=0.001,vmin=0,vmax=None,cmap="CMRmap",regulator="def",portion=0.999):
	drawfield=np.mean(field[:,:,zrange[0]:zrange[1]],axis=2)
	maxi=np.max(drawfield)
	mini=np.min(drawfield)
	if spanner==np.log:
		drawfield=(drawfield-mini+logbiaser)/(maxi-mini)
	else:
		drawfield=(drawfield-mini)/(maxi-mini)
	#Ignore extremal values by regulator
	if regulator=="def":
		maxplot=findportionrange(portion,drawfield,binnum=10000)
		#print(maxplot)
	elif regulator==None:
		maxplot=None
		pass
	else:
		maxplot=regulator(drawfield)
	fig=plt.figure(figsize=(10,10),dpi=100)
	if vmax==None:
		plt.imshow(spanner(drawfield),cmap=cmap,vmin=vmin,vmax=maxplot)
	else:
		plt.imshow(spanner(drawfield),cmap=cmap,vmin=vmin,vmax=vmax)

def summarize(filename,showmassf=True,size=250):
	dat=np.load(filename)
	xdat=dat["xdat"]
	vdat=dat["vdat"]
	mdat=dat["mdat"]
	print("Octant "+filename[-7:-4]+" of side "+str(size)+" Mpc containing "+str(len(xdat))+" particles.")
	print("Mean Density[10^10 M_sun h^2/(Mpc)^3]: "+str(np.sum(mdat)/size**3))
	print("V_rms[km/s]: "+str(np.sqrt(np.mean(np.square(vdat)))))
	if showmassf:
		_=fasthist(mdat)
		plt.xscale("log")
		plt.xlabel("Halo Mass")
		plt.ylabel("Count")
		plt.title("Halo Mass Function")
	if len(xdat)==len(vdat) and len(vdat)==len(mdat):
		print("No defects found")
	else:
		print("File Defected")
	return xdat,vdat,mdat

def simpfieldplot(field,zrange=[20,30],vmin=0,vmax=100,log=False,cmap="CMRmap"):
	drawfield=np.mean(field[:,:,zrange[0]:zrange[1]],axis=2)
	maxi=np.max(drawfield)
	mini=np.min(drawfield)
	drawfield=(drawfield-mini)/(maxi-mini)
	fig=plt.figure(figsize=(10,10),dpi=100)
	if log:
		plt.imshow(np.log(drawfield+1),cmap=cmap,vmin=vmin,vmax=vmax)
	else:
		plt.imshow(drawfield,cmap=cmap,vmin=vmin,vmax=vmax)

def maketree(dat,halfsize,mdat=None):
	first=True
	readpartnum=len(dat)
	print("Constructing tree from "+str(readpartnum)+" particles.")
	st=time.time()
	if mdat==None:
		mdat=np.ones(readpartnum)
	for i,(particle,m) in enumerate(zip(dat,mdat)):
		if first:
			tree=parttree3d(m,particle,halfsize,np.array([0.,0.,0.]))
			first=False
			continue
		tree.add_particle(m,particle)
		if readpartnum>100 and i%(readpartnum//100)==0 and i!=0:
			print("\r About "+str(i//(readpartnum/100)+1)+"% done over "+str(readpartnum)+" particles.",end='')
	formattedxdat=[[],[]]
	tree.write_data(formattedxdat)
	print(" ")
	print("Tree constructed. Time: "+str(time.time()-st))
	return tree,overlapped,np.array(formattedxdat[0]),np.array(formattedxdat[1])

def maketree(dat,halfsize,mdat=None,origin=np.array([0.,0.,0.])):
	first=True
	readpartnum=len(dat)
	print("Constructing tree from "+str(readpartnum)+" particles.")
	st=time.time()
	if mdat==None:
		mdat=np.ones(readpartnum)
	for i,(particle,m) in enumerate(zip(dat,mdat)):
		if first:
			tree=parttree3d(m,particle,halfsize,origin)
			first=False
			continue
		tree.add_particle(m,particle)
		if readpartnum>100 and i%(readpartnum//100)==0 and i!=0:
			print("\r About "+str(i//(readpartnum/100)+1)+"% done over "+str(readpartnum)+" particles.",end='')
	formattedxdat=[[],[]]
	tree.write_data(formattedxdat)
	print(" ")
	print("Tree constructed. Time: "+str(time.time()-st))
	return tree,overlapped,np.array(formattedxdat[0])-origin,np.array(formattedxdat[1])

def make2dtree(dat,halfsize):
	first=True
	readpartnum=len(dat)
	print("Constructing tree from "+str(readpartnum)+" particles.")
	st=time.time()
	for i,particle in enumerate(dat):
		if first:
			tree=parttree2d(1,particle,halfsize,np.array([0.,0.]))
			first=False
			continue
		tree.add_particle(1,particle)
		if i%(readpartnum//100)==0 and i!=0:
			print("\r About "+str(i//(readpartnum/100)+1)+"% done over "+str(readpartnum)+" particles.",end='')
	formattedxdat=[[],[]]
	tree.write_data(formattedxdat)
	print(" ")
	print("Tree constructed. Time: "+str(time.time()-st))
	return tree,overlapped,np.array(formattedxdat[0]),np.array(formattedxdat[1])

def checkoverlapped(overlapped,totpartnum=-1,formattedxdatlen=-1):
	print("Total "+str(len(overlapped))+" particles were glued due to overlap. Any which are false identifications: ")
	for overlap in overlapped:
		if overlap[0][0]!=overlap[1][0] or overlap[0][1]!=overlap[1][1] or overlap[0][2]!=overlap[1][2]:
			print("Not an overlap at: "+str(overlap[0]))
	print("")
	if formattedxdatlen!=-1 and totpartnum!=-1:
		if formattedxdatlen+len(overlapped)!=totpartnum:
			print(str(totpartnum-formattedxdatlen-len(overlapped))+" particles are lost")
		else:
			print("No particles lost")
	return

def fasthist(field,binnum=1000):
    hist,bins=np.histogram(field,binnum)
    meanbins=(bins[1:]+bins[:-1])/2
    plt.plot(meanbins,hist)
    plt.yscale("log")
    return hist, meanbins

def findportionrange(portion,field,binnum=2000,mini=0.):
	hist,bins=np.histogram(field,binnum)
	meanbins=(bins[1:]+bins[:-1])/2
	totnum=len(field.flatten())
	i=0
	cumm=0
	while(cumm<totnum*portion):
		if bins[i]<mini:
			i+=1
			continue
		cumm+=hist[i-1]
		i+=1
	return bins[i]

def stripext(field,bot=0,top=100):
    sh=field.shape
    tsize=np.prod(sh)
    edit=np.empty(tsize)
    ts=0
    bs=0
    print("Stripping "+str(tsize)+" values.")
    for i,el in enumerate(field.flatten()):
        if el>top:
        	edit[i]=top
        	ts+=1
        elif el<bot:
        	edit[i]=bot
        	bs+=1
        else:
        	edit[i]=el
    print(str(ts)+"("+str(100*ts/tsize)+"%) stripped to "+str(top)+" and "+str(bs)+"("+str(100*bs/tsize)+"%)"+" stripped to "+str(bot))
    return np.array(edit).reshape(sh)

def savefield(field,name=None):
	if name==None:
		name=raw_input("File path: ")
	np.save(name,field)


def genPoissonian(n=1000,size=1000,dim=3):
	dat=[]
	for d in range(dim):
		dat.append(np.random.rand(n)*size)
	return np.array(dat).T

def genShellDoublePoissonian(n1=50,n2=200,size=1000,shellmin=170,shellmax=200,dim=3):
	tempdat=[]
	for i in range(n1):
		cent=np.random.rand(dim)*size
		r=np.random.rand(n2)*(shellmax-shellmin)+shellmin
		theta=np.arccos(2*np.random.rand(n2)-1)
		phi=np.random.rand(n2)*2*np.pi
		if dim==3:
			pts=np.array([r*np.sin(theta)*np.cos(phi),r*np.sin(theta)*np.sin(phi),r*np.cos(theta)]).T+cent
		if dim==2:
			pts=np.array([r*np.sin(theta),r*np.sin(theta)]).T+cent
		tempdat.extend(pts)
	dat=[]
	for pt in tempdat:
		if dim==3:
			if pt[0]<=0 or pt[0]>=size or pt[1]<=0 or pt[1]>=size or pt[2]<=0 or pt[2]>=size:
				continue
		if dim==2:
			if pt[0]<=0 or pt[0]>=size or pt[1]<=0 or pt[1]>=size:
				continue		
		else:
			dat.append(pt)
	print(str(len(dat))+" points generated in box")
	return np.array(dat)

def scatter(points):
	fig=plt.figure(figsize=(10,10))
	plt.scatter(*points.T,s=1)

def scatter3D(points):
	fig = plt.figure(figsize=(8,8))
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(*points.T,s=1)

def plot2dgeo(polygonverts,circlex=np.array([0,0]),circler=1):
	fig=plt.figure(figsize=(6,6))
	ax = fig.subplots()
	ax.scatter(*polygonverts.T,c="red")
	for ed in verttoedges(polygonverts):
		ax.plot(*ed.T,c="orange")
	circle1 = plt.Circle(circlex, circler, color='b',fill=False)
	ax.add_artist(circle1)



def plotVoronoi(vor,points=None,highlight=None,xmax=100,ymax=100):
	fig=plt.figure(figsize=(10,10))
	ax=fig.add_subplot(111)
	sp.voronoi_plot_2d(vor,ax)
	if not highlight==None:
		ax.scatter(*points[highlight].T,s=100,c="red")
		ax.scatter(*vor.vertices[vor.regions[vor.point_region[highlight]]].T,s=100,c="green")
	plt.xlim(0,xmax)
	plt.ylim(0,ymax)


def DelaunayPlot(tri):
	fig=plt.figure(figsize=(10,10))
	ax=fig.add_subplot(111)
	sp.delaunay_plot_2d(tri,ax)

def VoronoiPlot(vor):
	fig=plt.figure(figsize=(10,10))
	ax=fig.add_subplot(111)
	sp.voronoi_plot_2d(vor,ax)

def minmaxmean(field):
    print(np.min(field),np.max(field),np.mean(field))

def inboxcheck(pts):
	for i in range(3):
	    print(np.min(pts[:,i]),np.max(pts[:,i]))



