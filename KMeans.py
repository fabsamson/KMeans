# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:34:52 2019

@author: Fabian
"""
import math
import numpy as np
import random
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class Point:
	def __init__(self,x,y,cluster=0):
		self.__x =x
		self.__y = y
		self.__cluster = cluster
	
	def getX(self):
		return self.__x
	
	def getY(self):
		return self.__y
	
	def getCluster(self):
		return self.__cluster
	
	def setCluster(self,cluster):
		self.__cluster = cluster
	
	def getCoords(self):
		return [self.__x,self.__y]
	

class KMeans:
	
	def __init__(self,nbClusters,max_iter = 500, distance = 'l2'):
		self.__nbClusters = nbClusters
		self.__max_iter = max_iter
		self.__distance = KMeans.__distance__(distance)
		self.__iter = 0
		self.__centroids = {}
		self.__initiated = False
		self.__hasChanged = True
		self.__points = []
	
	def getPoints(self):
		return self.__points
	
	def compute_next(self):
		random.shuffle(self.__points)
		for point in self.__points:
			dist = sys.maxsize
			clust = 0
			init_clust = point.getCluster()
			for i in range(1, self.__nbClusters+1):
				dist_i = self.computeDist(self.__centroids.get(i)[0].getCoords(),point.getCoords())
				if(dist_i < dist):
					dist = dist_i
					clust = i
			if(clust != init_clust):
				self.__hasChanged=True
				if(init_clust != 0):
					self.__centroids.get(init_clust)[1].remove(point)
					self.computeCentroid(self.__centroids.get(init_clust)[1],init_clust)
				self.__centroids.get(clust)[1].append(point)
				self.computeCentroid(self.__centroids.get(clust)[1],clust)
				point.setCluster(clust)
	
	def init(self):
		data = self.__points
		if(len(data) < self.__nbClusters):
			raise Exception("There are less data than clusters")
		random.shuffle(data)
		for i in range(1,self.__nbClusters+1):
			centroid = data[-(i)]
			centroid.setCluster(i)
			self.__centroids.update({i:[centroid,[centroid]]})
	
	def __distance__(dist):
		if(dist =='l2'):
			def distance(x1,x2):
				return math.sqrt(math.pow(x1[0]-x2[0],2)+math.pow(x1[1]-x2[1],2))
			return distance
		else:
			return dist
	
	def fit(self,data):
		for point in data:
			#print(point[0])
			self.__points.append(Point(point[0],point[1],0))
		if(not self.__initiated):
			self.init()
			self.__initiated = True
		while(self.__hasChanged and self.__iter < self.__max_iter):
			self.__hasChanged = False
			self.__iter +=1
			print("iter : "+str(self.__iter))
			self.compute_next()
	
	def computeDist(self,x1,x2):
		return self.__distance(x1,x2)
	
	def computeCentroid(self,centroid_data,cluster):
		size = 0
		x = 0
		y = 0
		for point in centroid_data:
			x += point.getX()
			y += point.getY()
			size +=1
		centroid = Point(x/size,y/size,cluster)
		self.__centroids.update({cluster:[centroid,self.__centroids.get(cluster)[1]]})
	
	def getResultsAsDF(self):
		data = []
		for point in self.getPoints():
			data.append([point.getX(),point.getY(),"Cluster "+str(point.getCluster())])
		data = pd.DataFrame(data,columns = ["x","y","cluster"])
		return data

kmeans = KMeans(5)

test_set = list(zip(np.random.randn(500),np.random.randn(500)))

kmeans.fit(test_set)

data = kmeans.getResultsAsDF()

sns.set(style="darkgrid")
sns.set_palette("muted")
g = sns.relplot(x='x',y='y',hue='cluster',data=data,s=100)
g.fig.suptitle("KMeans Results", x=0.5, y=0.98)
plt.subplots_adjust(top=0.90)