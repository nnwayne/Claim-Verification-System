from whoosh.fields import Schema, TEXT, STORED
import os.path,os,nltk,time,time,json,nltk,requests,gzip
from whoosh.index import open_dir
from whoosh.index import create_in
from whoosh.query import *
from whoosh.qparser import QueryParser,FuzzyTermPlugin,SequencePlugin,MultifieldPlugin
from nltk.tag.stanford import StanfordNERTagger
from whoosh import scoring
from pathlib import Path
from collections import Counter
from math import log, sqrt
from nltk.corpus import wordnet as wn
from collections import defaultdict,Counter

#set index
def SetIndex(indexName):
    schema = Schema(title=TEXT(stored=True), content=STORED, sent=STORED)
    if not os.path.exists(indexName):
        os.mkdir(indexName)
    ix = create_in(indexName, schema)
    ix = open_dir(indexName)
    filePath = "wiki-pages-text"
    pathDir=os.listdir(filePath)
    i=0
    # commits after reading every txt
    for allDir in pathDir:
        writer = ix.writer()
        importDoc(writer,allDir)
        writer.commit()
        i += 1
        print(i)
    print("importDocFinish")

#import the document in wiki
def importDoc(writer,allDir):
    child = "wiki-pages-text" + '/' + allDir
    fopen=open(child,'r')
    for lines in fopen.readlines():
        SLines = lines.split(" ")
        newTitle = SLines[0].replace("_"," ")
        linecontent = lines[len(SLines[0])+len(SLines[1])+2:]
        writer.add_document(title=newTitle, content=linecontent, sent=SLines[1])

SetIndex("WikiSplit")