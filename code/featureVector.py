__author__ = 'imjalpreet'

import re
from replaceRepeatingChars import replaceTwoOrMore

def getStopWordList(stopWordListFP):

    stopWords = []
    stopWords.append('||T||')
    stopWords.append('||U||')

    fp = open(stopWordListFP, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords

def getFeatureVector(tweet, stopWords):

    featureVector = []

    words = tweet.split()

    for w in words:
        w = replaceTwoOrMore(w)
        """
        strip punctuation
        """
        w = w.strip('\'"?,.')

        """
        check if the word stats with an alphabet
        """
        checkAlpha = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)

        """
        Remove stopWords
        """
        if w in stopWords or checkAlpha is None:
            continue
        else:
            featureVector.append(w.lower())

    return featureVector
