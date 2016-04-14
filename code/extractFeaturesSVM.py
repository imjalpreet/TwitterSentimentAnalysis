__author__ = 'imjalpreet'

import csv
import sys
from svmutil import *
from preProcess import preProcessTweet
from featureVector import getFeatureVector, getStopWordList, getAcronymList, getEmoticonList, getNegativeWords

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
        elif(tweetSentiment == 'neutral' or tweetSentiment == 'objective-OR-neutral' or tweetSentiment == 'objective'):
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

trainingTweets = csv.reader(open('data/training.tsv', 'rb'), delimiter='\t')
stopWords = getStopWordList('code/stopwords.txt')
acronyms = getAcronymList('data/InternetSlangAcronyms.txt')
emoticons = getEmoticonList('data/EmoticonSentimentLexicon.txt')
negativeWords = getNegativeWords('data/negativeWords.txt')
classifierDumpFile = 'classifierDump'
featureList = []

tweets = []
for row in trainingTweets:
    sentiment = row[2]

    """
    Un-comment for only positive and negative tweets
    """
    # if(sentiment == 'neutral' or sentiment == 'objective-OR-neutral' or sentiment == 'objective'):
    #     continue

    tweet = row[3]
    if tweet == 'Not Available':
        continue
    numberOfHashTags = tweet.count('#')
    processedTweet = preProcessTweet(tweet)
    featureVector = getFeatureVector(processedTweet, stopWords, acronyms, emoticons, negativeWords)
    featureVector.append('hash'+str(numberOfHashTags))
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

if sys.argv[1] == 'file':
    testingTweets = csv.reader(open('data/test.tsv', 'rb'), delimiter='\t')

    for row in testingTweets:
        sentiment = row[2]

        """
        Un-comment for only positive and negative tweets
        """
        # if(sentiment == 'neutral' or sentiment == 'objective-OR-neutral' or sentiment == 'objective'):
        #     continue

        tweet = row[3]
        if tweet == 'Not Available':
            continue
        numberOfHashTags = tweet.count('#')
        processedTweet = preProcessTweet(tweet)
        featureVector = getFeatureVector(processedTweet, stopWords, acronyms, emoticons, negativeWords)
        featureVector.append('hash'+str(numberOfHashTags))
        testTweets.append((featureVector, sentiment))

    """
    Test the classifier
    """
    test_feature_vector = getSVMFeatureVector(testTweets, featureList)

    favourable = []

    for (t, l) in testTweets:
        if l == 'positive':
            favourable.append(0)
        elif l == 'negative':
            favourable.append(1)
        else:
            favourable.append(2)

    """
    p_labels contains the final labeling result
    """
    p_labels, p_accs, p_vals = svm_predict(favourable, test_feature_vector, classifier)

elif sys.argv[1] == 'input':
    testTweet = raw_input("Please enter the Tweet: ")
    while testTweet != 'Exit':
        sentiment = raw_input("Please enter the expected sentiment: ")
        processedTweet = preProcessTweet(testTweet)
        featureVector = getFeatureVector(processedTweet, stopWords, acronyms, emoticons, negativeWords)
        testTweets.append((featureVector, sentiment))

        """
        Test the classifier
        """
        test_feature_vector = getSVMFeatureVector(testTweets, featureList)

        favourable = []

        for (t, l) in testTweets:
            if l == 'positive':
                favourable.append(0)
            elif l == 'negative':
                favourable.append(1)
            else:
                favourable.append(2)

        """
        p_labels contains the final labeling result
        """
        p_labels, p_accs, p_vals = svm_predict(favourable, test_feature_vector, classifier)

        testTweet = raw_input("Please enter the Tweet: ")
