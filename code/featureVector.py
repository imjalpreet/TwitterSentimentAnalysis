__author__ = 'imjalpreet'

import re
from replaceRepeatingChars import replaceTwoOrMore
from nltk import PorterStemmer
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures

stemmer = PorterStemmer()

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

def getEmoticonList(emoticonListFP):
    emoticonList = {}

    fp = open(emoticonListFP, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        word = word.split('\t')
        emoticonList[word[0]] = word[1]
        line = fp.readline()
    fp.close()
    return emoticonList

def getAcronymList(acronymListFP):
    acronymList = {}

    fp = open(acronymListFP, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        word = word.split('\t')
        acronymList[word[0]] = word[1]
        line = fp.readline()
    fp.close()
    return acronymList

def getNegativeWords(negativeListFP):
    negativeWords = []

    fp = open(negativeListFP, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        negativeWords.append(word)
        line = fp.readline()
    fp.close()
    return negativeWords

def getFeatureVector(tweet, stopWords, acronyms, emoticons, negativeWords):

    featureVector = []

    words = tweet.split()

    finalWords = []

    for word in range(len(words)):
        if acronyms.has_key(words[word]):
            words[word] = acronyms[words[word]].split(' ')
            finalWords.extend(words[word])
        elif emoticons.has_key(words[word]):
            words[word] = 'emo' + emoticons[words[word]]
            finalWords.append(words[word])
        elif words[word] in negativeWords:
            finalWords.append("negative")
        else:
            finalWords.append(words[word])

    for word in range(len(finalWords)):
        try:
            finalWords[word] = str(stemmer.stem(finalWords[word]))
        except:
            pass

    for word in finalWords:

        word = replaceTwoOrMore(word)
        """
        strip punctuation
        """
        word = word.strip('\'"?,.;&(){}[]!:-')

        """
        check if the word stats with an alphabet
        """
        checkAlpha = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", word)

        """
        Remove stopWords
        """
        if word in stopWords or checkAlpha is None:
            continue
        else:
            featureVector.append(word.lower())

    return featureVector

    """
    Un-comment for Bigrams
    """
    # findBigrams = BigramCollocationFinder.from_words(finalWords)
    # scoreFunction = BigramAssocMeasures.chi_sq
    # bigrams = findBigrams.nbest(scoreFunction, 20)
    #
    # for bigram in bigrams:
    #     firstWord = bigram[0].lower()
    #     secondWord = bigram[1].lower()
    #
    #     firstWord = replaceTwoOrMore(firstWord)
    #     secondWord = replaceTwoOrMore(secondWord)
    #
    #     firstWord = firstWord.strip('\'"?,.;&(){}[]!:-')
    #     secondWord = secondWord.strip('\'"?,.;&(){}[]!:-')
    #
    #     checkAlpha1 = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", firstWord)
    #     checkAlpha2 = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", secondWord)
    #
    #     if (firstWord in stopWords or secondWord in stopWords) or (checkAlpha1 is None or checkAlpha2 is None):
    #         continue
    #     else:
    #         Bigram = firstWord + ' ' + secondWord
    #         featureVector.append(Bigram)
    #
    # return featureVector

    """
    Un-comment for Trigrams
    """
    # findTrigrams = TrigramCollocationFinder.from_words(finalWords)
    # scoreFunction = TrigramAssocMeasures.chi_sq
    # trigrams = findTrigrams.nbest(scoreFunction, 20)
    #
    # for trigram in trigrams:
    #     firstWord = trigram[0].lower()
    #     secondWord = trigram[1].lower()
    #     thirdWord = trigram[2].lower()
    #
    #     firstWord = replaceTwoOrMore(firstWord)
    #     secondWord = replaceTwoOrMore(secondWord)
    #     thirdWord = replaceTwoOrMore(thirdWord)
    #
    #     firstWord = firstWord.strip('\'"?,.;&(){}[]!:-')
    #     secondWord = secondWord.strip('\'"?,.;&(){}[]!:-')
    #     thirdWord = thirdWord.strip('\'"?,.;&(){}[]!:-')
    #
    #     checkAlpha1 = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", firstWord)
    #     checkAlpha2 = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", secondWord)
    #     checkAlpha3 = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", thirdWord)
    #
    #     if (firstWord in stopWords or secondWord in stopWords or thirdWord in stopWords) or (checkAlpha1 is None or checkAlpha2 is None or checkAlpha3 is None):
    #         continue
    #     else:
    #         Trigram = firstWord + ' ' + secondWord + ' ' + thirdWord
    #         featureVector.append(Trigram)
    #
    # return featureVector
