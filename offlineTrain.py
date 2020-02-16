#!/usr/bin/python
# -*- coding:UTF-8 -*-

from ORF import OnlineRandomForest
import numpy as np

if __name__ == "__main__":
    orf = OnlineRandomForest(numTrees=100,input=722,pct=1)
    orf.offline_train()
