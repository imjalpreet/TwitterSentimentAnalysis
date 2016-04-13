__author__ = 'imjalpreet'

import csv
import pickle
import nltk
from preProcess import preProcessTweet
from featureVector import getFeatureVector, getStopWordList, getAcronymList, getEmoticonList

def extractFeatures(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features

trainingTweets = csv.reader(open('data/training.tsv', 'rb'), delimiter='\t')
stopWords = getStopWordList('code/stopwords.txt')
acronyms = getAcronymList('data/InternetSlangAcronyms.txt')
emoticons = getEmoticonList('data/EmoticonSentimentLexicon.txt')
classifierDump = 'classifierDumpNB'
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
    processedTweet = preProcessTweet(tweet)
    featureVector = getFeatureVector(processedTweet, stopWords, acronyms, emoticons)
    featureList.extend(featureVector)
    tweets.append((featureVector, sentiment))

"""
Remove featureList duplicates
"""
featureList = list(set(featureList))

"""
Extract feature vector for all tweets
"""
training_set = nltk.classify.util.apply_features(extractFeatures, tweets)

"""
Train the Naive Bayes Classifier
"""
fp = None
try:
    fp = open(classifierDump)
except:
    pass
if fp:
    NBClassifier = pickle.load(fp)
else:
    NBClassifier = nltk.NaiveBayesClassifier.train(training_set)
    classifierDumpFile = open(classifierDump, 'wb')
    pickle.dump(NBClassifier, classifierDumpFile)
    classifierDumpFile.close()

testingTweets = csv.reader(open('data/test.tsv', 'rb'), delimiter='\t')

testTweets = []
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
    processedTweet = preProcessTweet(tweet)
    testTweets.append((processedTweet, row[2]))

"""
Test the Classifier
"""
correct = 0
count = 0
for (testTweet, l) in testTweets:
    label = NBClassifier.classify(extractFeatures(getFeatureVector(testTweet, stopWords, acronyms, emoticons)))
    if label == l:
        correct += 1
    count += 1
print correct*100.0/count
print NBClassifier.show_most_informative_features(10)
