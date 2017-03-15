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

app = Flask(__name__)




MONGODB_HOST = 'localhost'
MONGODB_PORT = 27017
DBS_NAME = 'crime'
COLLECTION_NAME = 'projects'
FIELDS = {'county': True, 'year': True, 'crmrte': True, 'prbarr': True, 'prbconv': True, 'prbpris': True,'avgsen': True,'density': True,'wcon': True,'wfir': True,'wser': True,'wmfg': True,'_id': False}


def cluster_points(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters

def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu   

def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))         	

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/crime/projects")
def donorschoose_projects():
    connection = MongoClient(MONGODB_HOST, MONGODB_PORT)
    collection = connection[DBS_NAME][COLLECTION_NAME]
    projects = collection.find(projection=FIELDS)
    json_projects = []
    for project in projects:
        json_projects.append(project)
    json_projects = json.dumps(json_projects, default=json_util.default)
    connection.close()
    return json_projects



proj_details=donorschoose_projects();
crime_data = pd.read_json(proj_details)
crime_dataframe= pd.DataFrame(crime_data)
testarray = ast.literal_eval(proj_details)
clusterObj= crime_data[['county','year','crmrte','prbarr','prbconv','prbpris','avgsen',
						'density','wcon','wfir','wser','wmfg']]
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
clustervar['wfir']= preprocessing.scale(clustervar['wfir'].astype('float64'))
clustervar['wser']= preprocessing.scale(clustervar['wser'].astype('float64'))
clustervar['wmfg']= preprocessing.scale(clustervar['wmfg'].astype('float64'))

clus_train = clustervar
print(clus_train)
def findSuitableK():
	clusters=range(1,7) 
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
	# plt.show()

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

def find_centers(X, K):
	oldmu = random.sample(testarray, 15)
	mu = random.sample(testarray,15)
	# print(mu)
	while not has_converged(mu, oldmu):
	        oldmu = mu
	        print("this")
	        # Assign all points in X to clusters
	        clusters = cluster_points(testarray, mu)
	        # Reevaluate centers
	        mu = reevaluate_centers(oldmu, clusters)
	# print(mu)         
	# return(mu, clusters)    

find_centers(testarray,15)


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5005,debug=True)