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
import matplotlib.pylab as plt
from scipy.spatial.distance import cdist
import collections
from collections import defaultdict
from sklearn.decomposition import PCA
import math

app = Flask(__name__)

MONGODB_HOST = 'localhost'
MONGODB_PORT = 27017
DBS_NAME = 'crime'
COLLECTION_NAME = 'projects'
FIELDS = {'county': True, 'year': True, 'crmrte': True, 'prbarr': True, 'prbconv': True, 'prbpris': True,'avgsen': True,'density': True,'wcon': True,'wtuc': True,'wtrd': True,'wfir': True,'wser': True,'wmfg': True,'taxpc': True,'pctmin': True,'wfed': True,'wsta': True,'wloc': True,'mix': True,'pctymle': True,'_id': False}

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
print(crime_data)
# testarray = ast.literal_eval(proj_details)
clusterObj= crime_data[['county','year','crmrte','prbarr','prbconv','prbpris','avgsen',
						'density','wcon','wtuc','wtrd','wfir','wser','wmfg','taxpc','pctmin','wfed','wsta','wloc','mix','pctymle']]
clustervar=clusterObj.copy()

clustervar['county']= preprocessing.scale(clustervar['county'].astype('float64'))
clustervar['year']= preprocessing.scale(clustervar['year'].astype('float64'))
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
# print(clus_train)
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
	
	count=0;
	for i in range(0,3):
		length = len(cluster_dict[i])
		cluster_sample[i]=random.sample(cluster_dict[i],length//3)
		for k in cluster_sample[i]:
			test= clus_train.iloc[[k]]
			df=df.append(clus_train.iloc[[k]],ignore_index=True)
			# df.iloc[[count]] = clus_train.iloc[[k]]
			count =count +1
	return df

sampled_dataFrame=sampleClusters()

pca = PCA(n_components=21)
pca.fit(sampled_dataFrame)
existing_2d = pca.transform(sampled_dataFrame)
# print(existing_2d)
existing_df_2d = pd.DataFrame(existing_2d)
existing_df_2d.index = sampled_dataFrame.index
existing_df_2d.columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13','PC14','PC15','PC16','PC17','PC18','PC19','PC20','PC21']
# existing_df_2d.columns = ['PC1','PC2','PC3']
existing_df_2d.head()
# print(pca.get_covariance())
# print(pca.components_)  
# print(pca.components_.T * math.sqrt(pca.explained_variance_))
# print(pca.explained_variance_)
# print(pca.explained_variance_ratio_.cumsum())

def screeplot(pca, standardised_values):
    y = np.std(pca.transform(standardised_values), axis=0)**2
    # print(y)
    x = np.arange(len(y)) + 1 
    # print(x)
    plt.plot(x, y, "o-") 
    plt.xticks(x, ["PC"+str(i) for i in x], rotation=60)
    plt.ylabel("Variance")
    plt.show()
    return np.array(y),np.array(x)
    # showScreeplot()

y,x =screeplot(pca, sampled_dataFrame)  

@app.route("/crime/screeplot")
def showScreeplot():
	# print(y.tolist())
	return render_template("screePlot.html",y=y.tolist(),x=x.tolist())   



# def find_centers(X, K):
# 	oldmu = random.sample(testarray, 15)
# 	mu = random.sample(testarray,15)
# 	# print(mu)
# 	while not has_converged(mu, oldmu):
# 	        oldmu = mu
# 	        print("this")
# 	        # Assign all points in X to clusters
# 	        clusters = cluster_points(testarray, mu)
# 	        # Reevaluate centers
# 	        mu = reevaluate_centers(oldmu, clusters)
# 	# print(mu)         
# 	# return(mu, clusters)    

# find_centers(testarray,15)


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5005,debug=True)