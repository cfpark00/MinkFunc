import numpy as np
import time
import scipy.spatial as sp
import scipy.signal as sig
import scipy
import matplotlib.pyplot as plt
import multiprocessing as mp
import os

print("lsspy.DTFE: This library takes N-Body data and estimates it as a field using a Delaunay Tesselation Field Estimator. A Oct-tree structure is implemented so that a hierarchical algorithm can be applied.\n")

def DFTE3Wrapper(dat):
	pass

def DelaunayPlot(tri):
	sp.delaunay_plot_2d(tri)

def Delaunay3d(dat):
	print("Making Delaunay3d")
	st=time.time()
	tri=sp.Delaunay(dat)
	print("Delaunay3d Made. Time: "+str(time.time()-st))
	return tri

def triArea(p):
	p=p[1:]-p[0]
	return np.abs(np.cross(p[0],p[1]))/2.

def tetVol(p):
	p=p[1:]-p[0]
	return np.abs(np.dot(p[0],np.cross(p[1],p[2])))/6.

def coortofieldarg(coor,ngrid,size):
    return (ngrid*coor/size).astype(int)

def DelaunayFieldEstimate2(pointweights,tri,ngrid,size):
	res=np.zeros((ngrid-1,ngrid-1))
	totnum=len(tri.simplices)
	trf=tri.transform
	for i in range(ngrid-1):
		print("\r"+str(i+1)+" line done over "+str(ngrid-1),end="")
		for j in range(ngrid-1):
			coord=np.array([float(i+0.5)*size/ngrid,float(j+0.5)*size/ngrid])
			ind=tri.find_simplex(coord)
			T=trf[ind][:2,:]
			r=trf[ind][2,:]
			c=T.dot((coord-r))
			cr=1-c[0]-c[1]
			val=np.array([*c,cr]).dot(pointweigths[tri.simplices[ind]])
			res[i,j]=val
	return res
"""
def DelaunayFieldEstimate3(pointweights,tri,ngrid,size):
	if type(ngrid)==int:
		ngrid=(ngrid,ngrid,ngrid)
	res=np.full(ngrid,-1.)
	trf=tri.transform
	print("Computing DTFE in 3D over "+str(ngrid)+" sided grid.")
	st=time.time()
	for i in range(ngrid[0]):
		if i%(ngrid[0]/100)==0 and i!=0:
			print("\r About "+str(i//(ngrid[0]/100)+1)+"% done over "+str(ngrid[0])+" yz-slabs.",end='')
		for j in range(ngrid[1]):
			for k in range(ngrid[2]):
				coord=np.array([float(i+0.5)*size/ngrid[0],float(j+0.5)*size/ngrid[1],float(k+0.5)*size/ngrid[2]])
				ind=int(tri.find_simplex(coord))
				if ind==-1:
					continue
				T=trf[ind,:3,:3]
				#print("T")
				#print(T)
				r=trf[ind,3,:]
				#print(r)
				#print(coord)
				#print(coord-r)
				c=T.dot((coord-r))
				cr=1-c[0]-c[1]-c[2]
				#print(cr,c[0],c[1],c[2],cr+np.sum(c))
				val=np.array([*c,cr]).dot(pointweights[tri.simplices[ind]])
				#print(val)
				if val<0:
					print("weird")
				res[i,j,k]=val
	print("DTFE completed. Time: "+str(time.time()-st))
	print()
	return res
"""

def DelaunayFieldEstimate3(points,pointweights,ngrid,size):
	print("Computing DTFE in 3D over "+str(ngrid)+" sided grid.")
	x=np.linspace(0,size,ngrid)
	y=np.linspace(0,size,ngrid)
	z=np.linspace(0,size,ngrid)
	x11,y11,z11=np.mgrid[0:128:1000j,0:128:1000j,0:128:1000j]
	st=time.time()
	res=scipy.interpolate.griddata(points,pointweights,(x11,y11,z11),fill_value=-1)
	print("DTFE completed. Time: "+str(time.time()-st))
	print()
	return res

def MultiprocessedDFTE3(pointweights,tri,ngrid,size):
	print("Multiprocessing on "+str(size)+" processes.")

def GetAreaAndAdjacency(points,tri):
	totnum=len(tri.simplices)
	areas=np.zeros(totnum)
	adjacency=[[] for _ in range(len(points))]
	for i,simplice in enumerate(tri.simplices):
		#print("\r"+str(i+1)+" simplices done over "+str(totnum)+" simplices.",end='')
		for p in simplice:
			adjacency[p].append(i)
		areas[i]=triArea(*points[simplice,:].T)
	print()
	return areas,np.array(adjacency)

def GetVolAndAdjacency(points,tri):
	totnum=len(tri.simplices)
	vols=np.zeros(totnum)
	adjacency=[[] for _ in range(len(points))]
	print("Computing volumes and Adjacency.")
	st=time.time()
	for i,simplice in enumerate(tri.simplices):
		if i%(totnum//100)==0 and i!=0:
			print("\r About "+str(i//(totnum/100)+1)+"% done over "+str(totnum)+" simplices.",end='')
		for p in simplice:
			adjacency[p].append(i)
		vols[i]=tetVol(points[simplice,:])
	print("")
	print("Volumes and Adjacency computed. Time: "+str(time.time()-st))
	print()
	return vols,np.array(adjacency)

def calcweights(adjacency,areas):
	weight=np.zeros(len(adjacency))
	for i,adjacentsimplices in enumerate(adjacency):
		for simp in adjacentsimplices:
			weight[i]+=areas[simp]
	return 1./weight

def calcweights3(adjacency,vols):
	weight=np.zeros(len(adjacency))
	print("Computing Inverse Volume weights")
	totnum=len(adjacency)
	st=time.time()
	for i,adjacentsimplices in enumerate(adjacency):
		if i%(totnum//100)==0 and i!=0:
			print("\r About "+str(i//(totnum/100)+1)+"% done over "+str(totnum)+" points.",end='')
		for simp in adjacentsimplices:
			weight[i]+=vols[simp]
		if weight[i]==0:
			weight[i]=-1
			print("Zero volume at simplice: "+str(i))
	print("")
	print("Weight Calculation Done. Time: "+str(time.time()-st))
	print()
	return 1./weight

def GetPointValues(points,tri,ms=None):
	totnum=len(tri.simplices)
	vols=np.zeros(totnum)
	weights=np.full(len(points),0.)
	print("Computing point weights.")
	st=time.time()
	for i,simplice in enumerate(tri.simplices):
		if i%(totnum//100)==0 and i!=0:
			print("\r About "+str(i//(totnum/100)+1)+"% done over "+str(totnum)+" simplices.",end='')
		vol=tetVol(points[simplice,:])
		for p in simplice:
			weights[p]+=vol
	print("")
	print("Volumes and Adjacency computed. Time: "+str(time.time()-st))
	print()
	print("Check for zero volume or outside values:")
	for i,wei in enumerate(weights):
		if wei<=0:
			print("Zero or negative volume at point: "+str(i)+" Val: "+str(wei))
			print("Set to negative 1")
			weights[i]=-1
	return 4./weights

def fieldplot(field,zrange=[20,30],spanner=np.array,logbiaser=0.001,vmin=0,cmap="CMRmap",regulator="def",portion=0.999):
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
	plt.imshow(spanner(drawfield),cmap=cmap,vmin=vmin,vmax=maxplot)
