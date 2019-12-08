import numpy as np
import time
import scipy.spatial as sp
import scipy.signal as sig
import scipy
import matplotlib.pyplot as plt
import multiprocessing as mp
import os

print("lsspy.CorrSp: This library computes all correlating Minkowski functional contribution pairs. Only to be used when dealing with sparce data.\n")

def dist(x1,x2):
    return np.sqrt(np.sum(np.square(x1-x2)))

def reclen(dic):
    count=0
    for key,val in dic.items():
        count+=len(val)
        #print(len(val))
    return count

class correldict:
    def __init__(self,points,r=5.,calc2p=True):
        self.npoints=[{},{}]
        self.norder=[0,0]
        self.pnum=len(points)
        self.r=r
        if calc2p:
            print("Calculating 2-point")
            self.npoints.append({})
            self.dist2={}
            for i in range(self.pnum):
                first=True
                for j in range(self.pnum):
                    if i>=j:
                        continue
                    d=dist(points[i],points[j])
                    if d<2*self.r:
                        if first:
                            self.npoints[2][i]=[]
                            self.dist2[i]=[]
                            first=False
                        self.npoints[2][i].append(j)
                        self.dist2[i].append(d)
            self.norder.append(reclen(self.npoints[2]))
            print("Two Points calculated: "+str(self.norder[2])+" points")
    def gendictorder(self,order):
        if len(self.npoints)<order:
            assert False, "Generate Lower order first"
        if len(self.npoints)>order:
            assert False, "Already Generated"
        self.npoints.append({})
        if order==3:
            for i,adjlist in self.npoints[order-1].items():
                for ad in adjlist:
                    if not ad in self.npoints[2]:
                        continue
                    for add in self.npoints[2][ad]:
                        if add in adjlist:
                            self.npoints[order][(i,ad,add)]=True
            self.norder.append(len(self.npoints[order]))
        elif order>3:
            for i,corr in self.npoints[order-1].items():
                if not i[order-2] in self.npoints[2]:
                    continue
                for val in self.npoints[2][i[order-2]]:
                    valid=True
                    for ii in i[:-1]:
                        if val not in self.npoints[2][ii]:
                            valid=False
                    if valid:
                        self.npoints[order][(*i,val)]=True
            self.norder.append(len(self.npoints[order]))  
        else:
            assert False, "Generate order bigger than 3"
        print("Order "+str(order)+" generation done: "+str(self.norder[order])+" points")

    def calcallorder(self):
        order=len(self.npoints)
        go=True
        while(go):
            self.gendictorder(order)
            if self.norder[order]==0:
                go=False
            order+=1

    def displayorder(self,order,points=None,all=True):
        if all:
            plt.scatter(*points.T,s=10)
        for key in self.npoints[order].keys():
            plt.scatter(*points[list(key)].T,s=10,c="red")

def Voronoi(points):
	return sp.Voronoi(points)

def plotVoronoi(vor,points=None,highlight=None,xmax=100,ymax=100):
	fig=plt.figure(figsize=(10,10))
	ax=fig.add_subplot(111)
	sp.voronoi_plot_2d(vor,ax)
	if not highlight==None:
		ax.scatter(*points[highlight].T,s=100,c="red")
		ax.scatter(*vor.vertices[vor.regions[vor.point_region[highlight]]].T,s=100,c="green")
	plt.xlim(0,xmax)
	plt.ylim(0,ymax)

#Need Update
def getintsec(verts,circcent,r):
    return
def tricircint(triverts,circcent,r):
    invs=[False,False,False]
    for i,vert in enumerate(triverts):
        if dist(vert,circcent)<r:
            invs[i]=True
    for i,inv in enumerate(invs):
        for j,inv2 in enumerate(invs):
            if inv!=inv2:
                getintsec()


def genPoissonian(n=1000,size=100,dim=2):
	dat=[]
	for d in range(dim):
		dat.append(np.random.rand(n)*size)
	return np.array(dat).T

def scatter(points):
	fig=plt.figure(figsize=(10,10))
	plt.scatter(*points.T,s=1)

def polvol2d(verts):
    n=len(verts)
    xys=verts.T
    return np.abs(np.dot(xys[0][:],np.roll(xys[1][:],-1))-np.dot(xys[1][:],np.roll(xys[0][:],-1)))/2
def polcirc2d(verts):
    return np.sum(np.sqrt(np.square(verts-np.roll(verts,1))))

def minpar2d(rs,x,verts,nnd=None):
	res=np.empty((*rs.shape,3))
	if nnd==None:
		assert False, "Nearest Neighbor not given"
	maxr=0
	for vert in verts:
		d=dist(x,vert)
		if d>maxr:
			maxr=d
	if maxr>141:
		assert False, "computation wrong"
	vol=polvol2d(verts)
	if vol>4*maxr**2:
		assert False, "computation wrong"
	#circum=polcirc2d(vert)
	for i,r in enumerate(rs):
		if r<nnd/2:
			res[i][0]=np.pi*r**2
			res[i][1]=2*np.pi*r
			res[i][2]=1
			continue
		if r>maxr:
			#if r==9.091 and x[0]<10.869 and x[0]>10.868 and x[1]<8.026 and x[1]>8.02:
				#print(vol)
			res[i][0]=vol
			res[i][1]=0
			res[i][2]=0
			continue
		A,L,V=compute2dALV(r,x,verts)
		res[i][0]=A
		res[i][1]=L
		res[i][2]=V
	return res

def minpar3d(rs,x,verts,nnd=None,size=5000):
	res=np.empty((*rs.shape,4))
	if nnd==None:
		assert False, "Nearest Neighbor not given"
	maxr=0
	for vert in verts:
		d=dist(x,vert)
		if d>maxr:
			maxr=d
	if maxr>size:
		assert False, "computation wrong"
	vol=polvol2d(verts)
	if vol>4*maxr**2:
		assert False, "computation wrong"
	#circum=polcirc2d(vert)
	for i,r in enumerate(rs):
		if r<nnd/2:
			res[i][0]=np.pi*r**2
			res[i][1]=2*np.pi*r
			res[i][2]=1
			continue
		if r>maxr:
			res[i][0]=vol
			res[i][1]=0
			res[i][2]=0
			continue
		compute2dALV(r,x,verttoedges(verts))
		res[i][0]=0
		res[i][1:]=0
	return res

def verttoedges(verts):
	edges=[]
	verts2=np.roll(verts,-1,axis=0)
	for i in range(len(verts)):
		edges.append(np.array([verts[i],verts2[i]]))
	return np.array(edges)



def kdt(points):
	return sp.cKDTree(points)

def nnd(i,kdt,points):
	return dist(points[i],points[kdt.query(points[i],k=2)[1][1]])

def MinkFunc2d(rs,points,vor,kdt,size):
	minparlist=[]
	st=time.time()
	totnum=len(points)
	for i,county in enumerate(vor.point_region):
		if i%(totnum//100)==0 and i!=0:
			print("\r"+str(100*(i+1)/totnum)+" % processed over "+str(totnum)+" particles.",end="")
		if -1 in vor.regions[county]:
			minparlist.append(np.full((*rs.shape,3),0))
			continue
		cont=False
		for vert in vor.vertices[vor.regions[county]]:
			if vert[0]<0 or vert[0]>size or vert[1]<0 or vert[1]>size:
				minparlist.append(np.full((*rs.shape,3),0))
				cont=True
				break
		if cont:
			continue
		nndd=nnd(i,kdt,points)
		minparlist.append(minpar2d(rs,points[i],vor.vertices[vor.regions[county]],nndd))
	print()
	print("Minkowski functionals for "+str(len(points))+" particles computed. Time: "+str(time.time()-st))
	return np.array(minparlist)
	

def MFplot(rs,MF,dim,size,summing=None):
	if summing==None:
		mfplot=MF.sum(axis=0)
		fig, axs = plt.subplots(dim+1,2)
		fig.set_size_inches(15,5*(dim+1))
		for i in range(dim+1):
			axs[i,0].scatter(rs,mfplot[:,i]/size**(dim-i),label=str(i+1)+" th Minkowski functional")
			axs[i,0].set_xlabel("R")
			axs[i,0].set_ylabel("Functional Values over L**d")
			axs[i,0].legend()
			axs[i,1].scatter(rs,mfplot[:,i]/rs**(dim-i),label=str(i+1)+" th Minkowski functional")
			axs[i,1].set_xlabel("R")
			axs[i,1].set_ylabel("Normalized Functional Values")
			axs[i,1].legend()


def genmcpt(n,verts):
    poin=[]
    for i in range(n):
        a=sorted(np.random.rand(2))
        s=a[0]
        t=a[1]
        poin.append(np.array([s*verts[0][0]+(t-s)*verts[1][0]+(1-t)*verts[2][0],s*verts[0][1]+(t-s)*verts[1][1]+(1-t)*verts[2][1]]))
    return np.array(poin)

def mcint(n,verts,circcent,r):
    ps=genmcpt(n,verts)
    count=0.
    for p in ps:
        if dist(p,circcent)<r:
            count+=1
    return triArea(verts)*count/n

def findints(x,a,b,c):
    if a>=b:
        return False, (a**2*x[1]-a*b*x[0]-b*c)/(a**2+b**2)
    else:
        return True, (b**2*x[0]-a*b*x[1]-a*c)/(a**2+b**2)

def getline_abc(edge):
    return edge[1][1]-edge[0][1],edge[0][0]-edge[1][0],edge[0][1]*(edge[1][0]-edge[0][0])-edge[0][0]*(edge[1][1]-edge[0][1])

def disttoedge(x,edge,det=False):
    a,b,c=getline_abc(edge)
    if det:
        xory,ints=findints(x,a,b,c)
        if xory:
            bb=sorted(edge.T[0])
            return np.abs(a*x[0]+b*x[1]+c)/np.sqrt(a**2+b**2),bb[0]<ints and ints<bb[1]
        bb=sorted(edge.T[1])
        return np.abs(a*x[0]+b*x[1]+c)/np.sqrt(a**2+b**2),bb[0]<ints and ints<bb[1]
    return np.abs(a*x[0]+b*x[1]+c)/np.sqrt(a**2+b**2)

def plot2dgeo(polygonverts,circlex=np.array([0,0]),circler=1):
	fig=plt.figure(figsize=(6,6))
	ax = fig.subplots()
	ax.scatter(*polygonverts.T,c="red")
	for ed in verttoedges(polygonverts):
		ax.plot(*ed.T,c="orange")
	#plt.plot([0,6],[0,6])
	circle1 = plt.Circle(circlex, circler, color='b',fill=False)
	ax.add_artist(circle1)

def compute2dALV(r,x,verts):
    A=0
    L=0
    V=0
    edges=verttoedges(verts)
    for edge in edges:
        if len(edge)!=2:
            assert False, "Weird Edges Passed"
        dte,det=disttoedge(x,edge,det=True)
        v1=edge[0]-x
        v2=edge[1]-x
        v1n=np.sqrt(np.sum(np.square(v1)))
        v2n=np.sqrt(np.sum(np.square(v2)))
        theta=np.arccos(np.dot(v1,v2)/(v1n*v2n))
        if theta<0:
            assert False
        if v1n<=v2n:
            ordered=True
        else:
            ordered=False
        if det:# The circle joins the segment first 
            if r<=dte:
                A+=theta*r**2/2.
                L+=theta*r
            elif r>=v1n and r>=v2n:
                area=v1n*v2n*np.sin(theta)/2.
                A+=area
            elif ordered:
                if r>v1n:
                    v21=edge[1]-edge[0]
                    v21n=np.sqrt(np.sum(np.square(v21)))
                    theta3=np.arcsin(dte/r)
                    theta2=np.arccos(np.dot(v21,v2)/(v2n*v21n))
                    thetaf=theta3-theta2
                    thetan=theta-thetaf
                    A+=thetaf*r**2/2.+v1n*r*np.sin(thetan)/2.
                    if thetaf*r<0:
                        assert False, str(thetaf)
                    L+=thetaf*r
                    V+=1                    
                else:
                    intr=np.sqrt(r**2-dte**2)
                    thetam=2*np.arccos(dte/r)
                    thetafn=theta-thetam
                    A+=intr*dte+thetafn*r**2/2.
                    if thetafn*r<0:
                        assert False, str(thetafn)
                    L+=thetafn*r
                    V+=2
            else:
                if r>v2n:
                    v12=edge[0]-edge[1]
                    v12n=np.sqrt(np.sum(np.square(v12)))
                    theta3=np.arcsin(dte/r)
                    theta2=np.arccos(np.dot(v12,v1)/(v1n*v12n))
                    thetaf=theta3-theta2
                    thetan=theta-thetaf
                    A+=thetaf*r**2/2.+v2n*r*np.sin(thetan)/2.
                    if thetaf*r<0:
                        assert False, str(thetaf)
                    L+=thetaf*r
                    V+=1                  
                else:
                    intr=np.sqrt(r**2-dte**2)
                    thetam=2*np.arccos(dte/r)
                    thetafn=theta-thetam
                    A+=intr*dte+thetafn*r**2/2.
                    if thetafn*r<0:
                        assert False, str(thetafn)
                    L+=thetafn*r
                    V+=2 
        else:# The circle meets a vertice first
            if ordered:
                if r<=v1n:
                    A+=theta*r**2/2.
                    L+=theta*r
                elif r<v2n:
                    v21=edge[1]-edge[0]
                    v21n=np.sqrt(np.sum(np.square(v21)))
                    theta3=np.arcsin(dte/r)
                    theta2=np.arccos(np.dot(v21,v2)/(v2n*v21n))
                    thetaf=theta3-theta2
                    thetan=theta-thetaf
                    A+=thetaf*r**2/2.+v1n*r*np.sin(thetan)/2.
                    if thetaf*r<0:
                        assert False, str(thetaf)
                    L+=thetaf*r
                    V+=1
                else:
                    area=v1n*v2n*np.sin(theta)/2.
                    A+=area
            else:
                if r<=v2n:
                    A+=theta*r**2/2.
                    L+=theta*r
                elif r<v1n:
                    v12=edge[0]-edge[1]
                    v12n=np.sqrt(np.sum(np.square(v12)))
                    theta3=np.arcsin(dte/r)
                    theta2=np.arccos(np.dot(v12,v1)/(v1n*v12n))
                    thetaf=theta3-theta2
                    thetan=theta-thetaf
                    A+=thetaf*r**2/2.+v2n*r*np.sin(thetan)/2.
                    if thetaf*r<0:
                        assert False, str(thetaf)
                    L+=thetaf*r
                    V+=1
                else:
                    area=v1n*v2n*np.sin(theta)/2.
                    A+=area
    return A,L,V







