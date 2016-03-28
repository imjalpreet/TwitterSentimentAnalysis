__author__ = 'imjalpreet'

import csv
import svm
from svmutil import *
from preProcess import preProcessTweet
from featureVector import getFeatureVector, getStopWordList

def getSVMFeatureVectorAndLabels(tweets, featureList):
    sortedFeatures = sorted(featureList)
    Map = {}
    featureVector = []
    labels = []
    for t in tweets:
        label = 0
        Map = {}

        """
        Initialize empty Map
        """
        for w in sortedFeatures:
            Map[w] = 0

        tweetWords = t[0]
        tweetSentiment = t[1]

        for word in tweetWords:
            """
            Set Map[word] to 1 if word exists
            """
            if word in Map:
                Map[word] = 1

        values = Map.values()
        featureVector.append(values)
        if(tweetSentiment == 'positive'):
            label = 0
        elif(tweetSentiment == 'negative'):
            label = 1
        elif(tweetSentiment == 'neutral'):
            label = 2
        labels.append(label)

    return {'featureVector': featureVector, 'labels': labels}

def getSVMFeatureVector(tweets, featureList):
    sortedFeatures = sorted(featureList)
    Map = {}
    featureVector = []
    for t in tweets:
        label = 0
        Map = {}

        """
        Initialize empty Map
        """
        for w in sortedFeatures:
            Map[w] = 0

        tweetWords = t[0]

        for word in tweetWords:
            """
            Set Map[word] to 1 if word exists
            """
            if word in Map:
                Map[word] = 1

        values = Map.values()
        featureVector.append(values)
    return featureVector

inpTweets = csv.reader(open('data/training.tsv', 'rb'), delimiter='\t')
stopWords = getStopWordList('code/stopwords.txt')
classifierDumpFile = 'classifierDump'
featureList = []

tweets = []
for row in inpTweets:
    sentiment = row[2]
    tweet = row[3]
    if tweet == 'Not Available':
        continue
    processedTweet = preProcessTweet(tweet)
    featureVector = getFeatureVector(processedTweet, stopWords)
    featureList.extend(featureVector)
    tweets.append((featureVector, sentiment))

"""
Remove featureList duplicates
"""
featureList = list(set(featureList))

"""
Train the classifier
"""
result = getSVMFeatureVectorAndLabels(tweets, featureList)
problem = svm_problem(result['labels'], result['featureVector'])
"""
'-q' option suppress console output
"""
param = svm_parameter('-q')
param.kernel_type = LINEAR
fp = None
try:
    fp = open(classifierDumpFile, 'r')
except:
    pass
if fp:
    classifier = svm_load_model(classifierDumpFile)
else:
    classifier = svm_train(problem, param)
    svm_save_model(classifierDumpFile, classifier)

testTweets = []

inpTweets = csv.reader(open('data/test.tsv', 'rb'), delimiter='\t')

for row in inpTweets:
    sentiment = row[2]
    tweet = row[3]
    if tweet == 'Not Available':
        continue
    processedTweet = preProcessTweet(tweet)
    featureVector = getFeatureVector(processedTweet, stopWords)
    testTweets.append((featureVector, sentiment))

"""
Test the classifier
"""
test_feature_vector = getSVMFeatureVector(testTweets, featureList)

"""
p_labels contains the final labeling result
"""
p_labels, p_accs, p_vals = svm_predict([0] * len(test_feature_vector),test_feature_vector, classifier)

count = 0
total, correct, wrong = 0, 0, 0
accuracy = 0.0
for (t, l) in tweets:
    label = p_labels[count]
    if label == 0:
        label = 'positive'
    elif label == 1:
        label = 'negative'
    elif label == 2:
        label = 'neutral'

    if label == l:
        correct += 1
    else:
        wrong += 1
    total += 1
    count += 1

accuracy = (float(correct)/total)*100
print accuracy