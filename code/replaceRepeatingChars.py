__author__ = 'imjalpreet'

import re

"""
Replace a sequence of repeating characters by three characters
"""

def replaceTwoOrMore(s):
    pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
    return pattern.sub(r"\1\1\1", s)
