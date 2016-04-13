__author__ = 'imjalpreet'

import re

def preProcessTweet(tweet):

    tweet = tweet.lower()

    """
    Convert www.* or https?://* to ||U||
    """
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '||U||', tweet)

    """
    Convert @username to ||T||
    """
    tweet = re.sub('@[^\s]+', '||T||', tweet)

    """
    Replace #word with word
    """
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

    """
    Remove additional white spaces
    """
    tweet = re.sub('[\s]+', ' ', tweet)

    return tweet
