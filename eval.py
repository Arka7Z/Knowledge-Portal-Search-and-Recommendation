import argparse
from os import path
import sys
import json
import numpy as np
import csv

def dcg(relevanceScores, k = 10, method=0):
    """
    Returns discounted cumulative gain (dcg)
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    relevanceScores = np.asfarray(relevanceScores)[:k]
    if relevanceScores.size:
        if method == 0:
            return relevanceScores[0] + np.sum(relevanceScores[1:] / np.log2(np.arange(2, relevanceScores.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0

def ndcgMax(relevanceScores, k=10, method=0):
    return dcg(sorted(relevanceScores, reverse=True), k, method)

def ndcg(relevanceScores, ndcgMax, k = 10, method=0):
    return dcg(relevanceScores, k, method) / ndcgMax

parser = argparse.ArgumentParser(description='Evaluate NDCG on search results')
parser.add_argument('results', type=str, help='Input file containing ranked search results')
args = parser.parse_args()

filePath = './data/'
fileName = args.results
resultFile = filePath + fileName
if not path.exists(resultFile):
    print('File doesnt exist in data folder')
    sys.exit()
    
queryList = ['converting text to speech', 'Big data', 'efficient estimation of word representations in vector space', 'natural language interface', 'reinforcement learning in video game']
queryToIdx = {queryList[i]:i for i in range(len(queryList))}
annotationDict = [{} for i in range(len(queryList))]    # + 1 for 1 based indexing in
with open('./data/annotations.qrel') as file:
    for line in file:
        lineString = line.split()
        qid = int(lineString[0])
        docno = lineString[2]
        relScore = int(lineString[-1])
        annotationDict[qid][docno] = relScore
        
ndcgMaxPerQuery = []
for i in range(len(queryList)):
    ndcgMaxPerQuery.append(ndcgMax(list(annotationDict[i].values())))
        
relevanceScores= []       
with open(resultFile, 'r') as file:
    for line in file:
        data = json.loads(line)
        result = data['result']
        for i in range(len(result)):
            result[i] = [annotationDict[i].get(ID, 0) for ID in result[i]]
        relevanceScores.append(result)
results = []
meanScores = []

for i in range(len(relevanceScores)):
    relScoresForThisMethod = relevanceScores[i]
    ndcgScoresMethod = []
    for q in range(len(queryList)):
        ndcgScoresMethod.append(ndcg(relScoresForThisMethod[q], ndcgMaxPerQuery[q]))
    results.append(ndcgScoresMethod)
    meanScores.append(np.mean(ndcgScoresMethod))
    
print('Mean NDCG Scores: ')
print(meanScores)
