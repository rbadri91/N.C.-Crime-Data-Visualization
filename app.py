from flask import Flask
from flask import render_template
from pymongo import MongoClient
import json
from bson import json_util
from bson.json_util import dumps
import random
import numpy as np
import ast
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist,squareform
import collections
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import pairwise_distances 
import math

app = Flask(__name__)

MONGODB_HOST = 'localhost'
MONGODB_PORT = 27017
DBS_NAME = 'crime'
COLLECTION_NAME = 'projects'
FIELDS = {'crmrte': True, 'prbarr': True, 'prbconv': True, 'prbpris': True,'avgsen': True,'density': True,'wcon': True,'wtuc': True,'wtrd': True,'wfir': True,'wser': True,'wmfg': True,'taxpc': True,'pctmin': True,'wfed': True,'wsta': True,'wloc': True,'mix': True,'pctymle': True,'_id': False}

@app.route("/")
def index():
    return render_template("index.html")   

@app.route("/crime/projects")
def crime_projects():
    connection = MongoClient(MONGODB_HOST, MONGODB_PORT)
    collection = connection[DBS_NAME][COLLECTION_NAME]
    projects = collection.find(projection=FIELDS)
    json_projects = []
    for project in projects:
        json_projects.append(project)
    json_projects = json.dumps(json_projects, default=json_util.default)
    connection.close()
    return json_projects



proj_details=crime_projects();
crime_data = pd.read_json(proj_details)  

# testarray = ast.literal_eval(proj_details)
clusterObj= crime_data[['crmrte','prbarr','prbconv','prbpris','avgsen',
						'density','wcon','wtuc','wtrd','wfir','wser','wmfg','taxpc','pctmin','wfed','wsta','wloc','mix','pctymle']]
clustervar=clusterObj.copy() 

# clustervar['county']= preprocessing.scale(clustervar['county'].astype('float64'))
# clustervar['year']= preprocessing.scale(clustervar['year'].astype('float64'))
clustervar['crmrte']= preprocessing.scale(clustervar['crmrte'].astype('float64'))
clustervar['prbarr']= preprocessing.scale(clustervar['prbarr'].astype('float64'))
clustervar['prbconv']= preprocessing.scale(clustervar['prbconv'].astype('float64'))
clustervar['prbpris']= preprocessing.scale(clustervar['prbpris'].astype('float64'))
clustervar['avgsen']= preprocessing.scale(clustervar['avgsen'].astype('float64'))
clustervar['density']= preprocessing.scale(clustervar['density'].astype('float64'))
clustervar['wcon']= preprocessing.scale(clustervar['wcon'].astype('float64'))
clustervar['wtuc']= preprocessing.scale(clustervar['wtuc'].astype('float64'))
clustervar['wtrd']= preprocessing.scale(clustervar['wtrd'].astype('float64'))
clustervar['wfir']= preprocessing.scale(clustervar['wfir'].astype('float64'))
clustervar['wser']= preprocessing.scale(clustervar['wser'].astype('float64'))
clustervar['wmfg']= preprocessing.scale(clustervar['wmfg'].astype('float64'))
clustervar['taxpc']= preprocessing.scale(clustervar['taxpc'].astype('float64'))
clustervar['pctmin']= preprocessing.scale(clustervar['pctmin'].astype('float64'))
clustervar['wfed']= preprocessing.scale(clustervar['wfed'].astype('float64'))
clustervar['wsta']= preprocessing.scale(clustervar['wsta'].astype('float64'))
clustervar['wloc']= preprocessing.scale(clustervar['wloc'].astype('float64'))
clustervar['mix']= preprocessing.scale(clustervar['mix'].astype('float64'))
clustervar['pctymle']= preprocessing.scale(clustervar['pctymle'].astype('float64'))   

clus_train = clustervar
def findSuitableK():
	clusters=range(1,9) 
	meandist=[]
	for k in clusters:
	    model=KMeans(n_clusters=k)
	    model.fit(clus_train)
	    clusassign=model.predict(clus_train)
	    meandist.append(sum(np.min(cdist(clus_train, model.cluster_centers_, 'euclidean'), axis=1))
	    / clus_train.shape[0])
	plt.plot(clusters, meandist)
	plt.xlabel('Number of clusters') 
	plt.ylabel('Average distance')
	plt.title('Selecting k with the Elbow Method') # pick the fewest number of clusters that reduces the average distance    
	plt.show()

findSuitableK()

def createClusters():
	model=KMeans(n_clusters=3)
	model.fit(clus_train)
	clusassign=model.predict(clus_train) 
	lables = model.labels_
	return lables

lables=createClusters()

def groupClusters():
	my_dict = {}
	for (ind,elem) in enumerate(lables):
		if elem in my_dict:
			my_dict[elem].append(ind)
		else:
			my_dict.update({elem:[ind]})
	return my_dict

cluster_dict=groupClusters()

def sampleClusters():
	cluster_sample={} 
	df = pd.DataFrame()
	# # df = pd.DataFrame(index=range(0,13),columns=['county','year','crmrte','prbarr','prbconv','prbpris','avgsen',
	# 					'density','wcon','wfir','wser','wmfg'], dtype='float64')
	# df= pd.DataFrame([['county','year','crmrte','prbarr','prbconv','prbpris','avgsen',
	# 					'density','wcon','wfir','wser','wmfg']])
	
	for i in range(0,3):
		length = len(cluster_dict[i])
		cluster_sample[i]=random.sample(cluster_dict[i],length//3)
		for k in cluster_sample[i]:
			df=df.append(clus_train.iloc[[k]],ignore_index=True)
			# df.iloc[[count]] = clus_train.iloc[[k]]
	return df

def randomSample():
	newClusterTrain= clustervar.sample(n=len(clus_train)//3)
	return newClusterTrain

randomSampledClusterFrame=randomSample()	

sampled_dataFrame=sampleClusters()

pca = PCA(n_components=19)
pca.fit(sampled_dataFrame)
loadings=pca.components_ 

def pcaRandomSample():
	r_pca = PCA(n_components=19) 
	r_pca.fit_transform(randomSampledClusterFrame)
	r_loadings=r_pca.components_
	return r_pca,r_loadings

random_pca,Random_loadings =pcaRandomSample()


def screeplot(pca, standardised_values):
    y = pca.explained_variance_
    x = np.arange(len(y)) + 1 
    plt.plot(x, y, "o-") 
    plt.xticks(x, ["PC"+str(i) for i in x], rotation=60) 
    plt.ylabel("Variance")
    plt.show()
    return np.array(y),np.array(x)

y,x =screeplot(pca, sampled_dataFrame) 
y_random,x_random =screeplot(random_pca, Random_loadings) 



@app.route("/crime/screeplot")
def showScreeplot():
	return render_template("screeplot.html",y=y.tolist(),x=x.tolist()) 

@app.route("/crime/randomscreeplot")
def showScreeplot_random():
	return render_template("randomScreePlot.html",y=y_random.tolist(),x=x_random.tolist()) 


def squaredLoadings():
	w, h = 3, 19;
	squaredLoadings = [0 for y in range(h)] 
	for i in range(len(loadings)):
		sum=0
		for j in range(3):  
			sum = sum + loadings[j][i] **2
		squaredLoadings[i]=sum	
	return squaredLoadings

sumSquareLoadings=squaredLoadings()

@app.route("/crime/squaredLoadings")
def showSqureloadingsPlot():
	sortedSumSquareLoadings=sorted(sumSquareLoadings,reverse=True)
	length= len(sortedSumSquareLoadings)
	columns=[0 for y in range(length)] 
	index=0
	for i in sortedSumSquareLoadings: 
		columns[index]=clus_train.columns.values[sumSquareLoadings.index(i)]
		index =index+1	
	return render_template("squaredloadings.html",y=sortedSumSquareLoadings,x=json.dumps(columns))


def randomSquaredLoadings():
	w, h = 3, 21;
	squaredLoadings = [0 for y in range(h)] 
	for i in range(len(Random_loadings)):
		sum=0
		for j in range(3):  
			sum = sum + loadings[j][i] **2
		squaredLoadings[i]=sum	
	return squaredLoadings

randomsumSquareLoadings=randomSquaredLoadings()

@app.route("/crime/randomsquaredLoadings")
def showRandomSqureloadingsPlot():
	sortedSumSquareLoadings=sorted(randomsumSquareLoadings,reverse=True)
	length= len(sortedSumSquareLoadings)
	columns=[0 for y in range(length)] 
	index=0
	for i in sortedSumSquareLoadings: 
		columns[index]=clus_train.columns.values[randomsumSquareLoadings.index(i)]
		index =index+1	
	return render_template("randomSquaredloadings.html",y=sortedSumSquareLoadings,x=json.dumps(columns))	

@app.route("/crime/scatterMatrix") 
def getColumnData():
	sortedSumSquareLoadings=sorted(sumSquareLoadings,reverse=True)
	columns=[0 for y in range(3)]	
	columnsVals={}
	index=0
	for i in sortedSumSquareLoadings:  
		columns[index]=clus_train.columns.values[sumSquareLoadings.index(i)] 
		index =index+1
		if index==3:
			break
	for i in range(3):
		columnsVals.update({columns[i]:sampled_dataFrame.loc[:,columns[i]].tolist()})
	return render_template("scatterMatrix.html",dataVal=columnsVals,traits=columns)	

@app.route("/crime/randomScatterMatrix") 
def getRandomColumnData():
	sortedSumSquareLoadings=sorted(randomsumSquareLoadings,reverse=True)
	columns=[0 for y in range(3)]	
	# columnsVals=[0 for y in range(3)]
	columnsVals={}
	index=0
	for i in sortedSumSquareLoadings:  
		columns[index]=clus_train.columns.values[randomsumSquareLoadings.index(i)] 
		index =index+1
		if index==3:
			break
	for i in range(3):
		columnsVals.update({columns[i]:randomSampledClusterFrame.loc[:,columns[i]].tolist()})
	return render_template("scatterMatrix.html",dataVal=columnsVals,traits=columns)	

	


def MDS_DimReduction():
	mdsData = MDS(n_components=2,dissimilarity='euclidean') 
	mdsData.fit(sampled_dataFrame)
	return mdsData.embedding_

def MDS_RandomDimReduction():
	mdsData = MDS(n_components=2,dissimilarity='euclidean') 
	mdsData.fit(randomSampledClusterFrame)
	return mdsData.embedding_

def PCA_TopComp():
	pca = PCA(n_components=2)
	return pca.fit_transform(sampled_dataFrame)

def PCA_RandomTopComp():
	pca = PCA(n_components=2)
	return pca.fit_transform(randomSampledClusterFrame)	

top_PCAVal=PCA_TopComp()
top_RandomPCAVal=PCA_RandomTopComp()

@app.route("/crime/scatterPlot")
def PCA_ScatterPlot():
	return render_template("scatterPlot.html",dataVal=top_PCAVal.tolist())

@app.route("/crime/randomscatterPlot")
def PCA_RandomScatterPlot():
	return render_template("randomScatterPlot.html",dataVal=top_RandomPCAVal.tolist())	

def MDS_DimReduction_Correlation(): 
	mdsData = MDS(n_components=2,dissimilarity='precomputed',max_iter=10)
	precompute=pairwise_distances(sampled_dataFrame.values,metric='correlation')
	return mdsData.fit_transform(precompute)

def MDS_Random_DimReduction_Correlation():  
	mdsData = MDS(n_components=2,dissimilarity='precomputed',max_iter=10)
	precompute=pairwise_distances(randomSampledClusterFrame.values,metric='correlation')
	return mdsData.fit_transform(precompute)	

@app.route("/crime/MDSscatterPlot")
def MDS_ScatterPlot():
	return render_template("ScatterPlotMDS.html",dataVal=mds_embeddings.tolist())

@app.route("/crime/MDSCorrelationscatterPlot")
def MDS_ScatterPlot_Correlation():
	return render_template("ScatterPlotMDS.html",dataVal=mds_embeddings_correlation.tolist())

@app.route("/crime/MDSRandomscatterPlot")
def MDS_RandomScatterPlot():
	return render_template("RandomScatterPlotMDS.html",dataVal=mds_RandomEmbeddings.tolist())

@app.route("/crime/MDSRandomCorrelationscatterPlot")
def MDS_RandomScatterPlot_Correlation():
	return render_template("RandomScatterPlotMDS.html",dataVal=mds_Random_embeddings_correlation.tolist())		

mds_embeddings=MDS_DimReduction()
mds_embeddings_correlation=MDS_DimReduction_Correlation()

mds_RandomEmbeddings=MDS_RandomDimReduction()
mds_Random_embeddings_correlation=MDS_Random_DimReduction_Correlation()

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5005,debug=True)