#!/usr/bin/python
# -*- coding:UTF-8 -*-

from ORF import OnlineRandomForest
import numpy as np

if __name__ == "__main__":
    orf = OnlineRandomForest(numTrees=1000,input=722,dataPath="./data/test/",pct=0)
    orf.test()
