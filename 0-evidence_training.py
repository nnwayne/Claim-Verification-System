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


def SetIndex(indexName):
    schema = Schema(title=TEXT(stored=True), content=STORED, sent=STORED)
    if not os.path.exists(indexName):
        os.mkdir(indexName)
    ix = create_in(indexName, schema)
    ix = open_dir(indexName)
    filePath = "wiki-pages-text"
    pathDir=os.listdir(filePath)
    # i=0
    # commits after reading every txt
    for allDir in pathDir:
        writer = ix.writer()
        importDoc(writer,allDir)
        writer.commit()
        # i += 1
        # print(i)
    print("importDocFinish")


def importDoc(writer,allDir):
    child = "wiki-pages-text" + '/' + allDir
    fopen=open(child,'r')
    for lines in fopen.readlines():
        SLines = lines.split(" ")
        newTitle = SLines[0].replace("_"," ")
        linecontent = lines[len(SLines[0])+len(SLines[1])+2:]
        writer.add_document(title=newTitle, content=linecontent, sent=SLines[1])


def tagSent2(TaggedClaim):
    # received tagged claim, and gather entities
    Entities = []
    length = len(TaggedClaim)
    lst = iter(range(length-1,-1,-1))
    for i in lst:
        if TaggedClaim[i][1] != 'O':
            EntityName = TaggedClaim[i][0]
            j=i-1
            while j>-1:
                if TaggedClaim[j][1] != 'O':
                    EntityName = TaggedClaim[j][0]+" "+EntityName
                    lst.__next__()
                    j -= 1
                else:
                    Entities.append(EntityName)
                    EntityName = ""
                    break
            if EntityName != "":
                Entities.append(EntityName)
                EntityName = ""
    return Entities


def tagSent(sent,OriEntities):
    # process claims using POS tag, and gather entities
    words=nltk.word_tokenize(sent)
    pos_tags =nltk.pos_tag(words)
    Entities = {}
    EntityName = ""
    Others = ""
    Noun = ""
    length = len(pos_tags)
    NounTag = ["NN","NNS","CD"]
    NNPTag = ["NNP","NN"]
    Entities["NNP"]=[]
    Entities["NOUN"]=[]
    FinalEntities = []
    lst = iter(range(length-2,-1,-1))
    for i in lst:
        if pos_tags[i][1] == 'NNP':
            InEn = False
            for entity in OriEntities:
                if pos_tags[i][0] in entity:
                    InEn = True
            if InEn == False:
                Entities["NNP"].append(pos_tags[i][0])
                # print(Entities)
            # j=i-1
            # while j>=0 and (pos_tags[j][1] in NNPTag):
            #     EntityName = pos_tags[j][0]+" "+EntityName
            #     lst.__next__()
            #     j -= 1
                    

        elif pos_tags[i][1] in NounTag:
            Noun = pos_tags[i][0]
            j=i-1
            while j>=0 and (pos_tags[j][1] == "JJ" or pos_tags[j][1] in NounTag):
                Noun = pos_tags[j][0]+" "+Noun
                lst.__next__()
                j -= 1
            Entities["NOUN"].append(Noun)

        elif Others == "":
            Others += pos_tags[i][0]
        else:
            Others = pos_tags[i][0]+" "+Others
    Entities["OTHER"] = Others
    if len(Entities["NNP"]) >0:
        for i in range(len(Entities["NNP"])-1,-1,-1):
        # for ent in Entities["NNP"]:
            FinalEntities.append(Entities["NNP"][i])
    elif len(Entities["NOUN"]) >0:
        for i in range(len(Entities["NOUN"])-1,-1,-1):
        # for ent in Entities["NNP"]:
            FinalEntities.append(Entities["NOUN"][i])

    return FinalEntities


def checkLargeWord(Entities):

    # these words have a large number of related titles
    # we should remove these words at the first step
    LargeWord = ['American', 'United States', 'English', 'England', 'America', 'Canadian', 
    'British', 'French', 'Canada', 'California', 'German', 'Indian', 'France', 'Boston', 
    'Chinese', 'London', 'Europe', 'Russia', 'Paris', 'Japanese', 'United Kingdom', 'New York', 
    'Spain', 'Austria', 'Colombia', 'Florida', 'Los Angeles','China', 'U.S.', 'O. J. Simpson', 
    'Brazil', 'UK', 'South Korea','Google', 'Star Wars', 'European', 'Ireland', 'Beautiful', 
    'Russian', 'San Francisco', 'Hawaii', 'Belgium', 'Chicago', 'Charles', 'Britain', 
    'Pennsylvania', 'film', 'Las Vegas', 'Italy', 'Italian', 'Indonesia', 'World War II', 'Liverpool', 
    'Academy Awards', 'Australian', 'Hollywood', 'Constantine', 'Singapore', 'Renaissance','Indiana']

    
        # if len(Entities) == 1 and Entities[0] in LargeWord:
        #     # weighting
    TempEntities = Entities
    if len(Entities) > 0:
        for ent in Entities:
            if ent in LargeWord:
                TempEntities.remove(ent)
            # if len(TempEntities) == 0:
            #     # weighting
    return TempEntities


def generateQuery(claim,TaggedClaim):
    query = ""
    qnnp = ""
    Entities = []

    # Entities = ["EntityName","EntityName",....]
    Entities = tagSent2(TaggedClaim)
    Entities += tagSent(claim,Entities)
    Entities = checkLargeWord(Entities)
    if len(Entities) != 0:
        for nnp in Entities:
            query = query+ "("+nnp+") OR "
    else:
        Entities = tagSent(claim,Entities)
        Entities = checkLargeWord(Entities)
        if len(Entities) != 0:
            for nnp in Entities:
                query = query+ "("+nnp+") OR "

    if query == "":
        query = generateQuerySecond(claim)
    return query


def generateQuerySecond(claim):
    query = ""

    words=nltk.word_tokenize(claim)
    pos_tags =nltk.pos_tag(words)

    NOUN = ["NN","NNS","CD","NNP"]

    for i in range(len(pos_tags)):
        # if pos_tags[i][1] in NOUN:
        if pos_tags[i][1] == "NNP":
            query = query+ "("+pos_tags[i][0]+") OR "
    if query == "":
        for i in range(len(pos_tags)):
        # if pos_tags[i][1] in NOUN:
            if pos_tags[i][1] in NOUN:
                query = query+ "("+pos_tags[i][0]+") OR "
    return query



def QuerySent(query):
    # only return the 100 most similar titles with theirs contents
    myquery = parser.parse(query)
    results =searcher.search(myquery,limit=None)
    return results


def readClaims(read_file):
    # count is used to control the number of claims that should be read
    claimTogether=""
    claimsWithId = []
    twolabel = ["SUPPORTS","REFUTES"]
    with open(read_file, "r") as r_file:
        trainDic = json.load(r_file)
        for key, value in trainDic.items():
            if value["label"] == "SUPPORTS":
                claimsWithId.append((key,value["claim"],value["evidence"]))
                claimTogether = claimTogether + value["claim"] + " SPLITATHERE "
    return claimTogether,claimsWithId


class InvertedIndex:
    def __init__(self, vocab, doc_term_freqs):
        self.vocab = vocab
        self.doc_len = [0] * len(doc_term_freqs)
        self.doc_term_freqs = [[] for i in range(len(vocab))]
        self.doc_ids = [[] for i in range(len(vocab))]
        self.doc_freqs = [0] * len(vocab)
        self.total_num_docs = 0
        self.max_doc_len = 0
        for docid, term_freqs in enumerate(doc_term_freqs):
            doc_len = sum(term_freqs.values())
            self.max_doc_len = max(doc_len, self.max_doc_len)
            self.doc_len[docid] = doc_len
            self.total_num_docs += 1
            for term, freq in term_freqs.items():
                term_id = vocab[term]
                self.doc_ids[term_id].append(docid)
                self.doc_term_freqs[term_id].append(freq)
                self.doc_freqs[term_id] += 1

    def num_terms(self):
        return len(self.doc_ids)

    def num_docs(self):
        return self.total_num_docs

    def docids(self, term):
        term_id = self.vocab[term]
        return self.doc_ids[term_id]

    def freqs(self, term):
        term_id = self.vocab[term]
        return self.doc_term_freqs[term_id]

    def f_t(self, term):
        term_id = self.vocab[term]
        return self.doc_freqs[term_id]

    def space_in_bytes(self):
        # this function assumes each integer is stored using 8 bytes
        space_usage = 0
        for doc_list in self.doc_ids:
            space_usage += len(doc_list) * 8
        for freq_list in self.doc_term_freqs:
            space_usage += len(freq_list) * 8
        return space_usage

# given a query and an index returns a list of the k highest scoring documents as tuples containing <docid,score>
def query_tfidf(query, index, vocab, k=10):

    # scores stores doc ids and their scores
    scores = Counter()

    N = index.num_docs()
    for i, d in enumerate(index.doc_len):
        s = 0
        for qterm in query:
            if qterm in vocab.keys():
                fdt = 0
                if i in index.docids(qterm):
                    docLoc = index.docids(qterm).index(i)
                    fdt = index.freqs(qterm)[docLoc]
                ft = index.f_t(qterm)
                s = log(1+fdt)+s
                # s = log(1+fdt)*log(N/ft)+s
        # s = s/sqrt(index.doc_len[i])
        scores[i] += s

    return scores.most_common(k)


def processResult(claim,raw_results):
    raw_docs = []
    sentNum = []
    trashWord = ['the','an','be','been','to','(',')','.',',','it','him','himself','herself','not']
    ori_claim = claim
    for rs in raw_results:
        allcon = rs['title']+" , "+rs['content']
        raw_docs.append(allcon)
        sentNum.append(rs['sent'])

    # processed_docs stores the list of processed docs
    processed_docs = []
    # vocab contains (term, term id) pairs
    vocab = {}

    stemmer = nltk.stem.PorterStemmer()

    for raw_doc in raw_docs:

        # norm_doc stores the normalized tokens of a doc
        norm_doc = []

        tokenized_doc = nltk.word_tokenize(raw_doc)
        norm_doc = [stemmer.stem(token.lower()) for token in tokenized_doc]
        for token in norm_doc:
            if token not in vocab:
                vocab[token] = len(vocab)
        # total_tokens = total_tokens+len(norm_doc)
        processed_docs.append(norm_doc)

    doc_term_freqs = []

    for processed_doc in processed_docs:
        a = Counter(processed_doc)
        doc_term_freqs.append(a)

    invindex = InvertedIndex(vocab, doc_term_freqs)
    # We output some statistics from our index
    stemmed_query = nltk.word_tokenize(nltk.stem.PorterStemmer().stem(claim))
    for word in trashWord:
        if word in stemmed_query:
            second_stemmed = list(filter(lambda tok: tok != word, stemmed_query))
    processed_results = query_tfidf(second_stemmed, invindex, vocab)
    Top10Res = []
    Top10SentsNum = []
    for rank, res in enumerate(processed_results):
    #     # e.g RANK 1 DOCID 176 SCORE 0.426 CONTENT South Korea rose 1% in February from a year earlier, the
    #     print("RANK {:2d} DOCID {:8d} SCORE {:.3f} CONTENT {:}".format(rank+1,res[0],res[1],raw_docs[res[0]]))
        Top10Res.append((raw_docs[res[0]],int(sentNum[res[0]])))
    return Top10Res



def getDifferentTag(sent):
    words=nltk.word_tokenize(sent)
    pos_tags =nltk.pos_tag(words)
    NounTag = ["NN","NNS"]
    NNPTag = ["NNP","CD","NNPS"]
    JJTag = ["JJ","JJR","JJS","RB","RBR","RBS"]
    VEBTag = ["VB","VBG","VBD","VBN","VBP","VBZ"]
    VEBTrash = ['is','was','are','were','has','have','had']
    NNPTrash = ["-LRB-","-LSB-","-RRB-","-RSB-"]
    TagDic={}
    TagDic["NNP"] = []
    TagDic["NOUN"] = []
    TagDic["JJ"] = []
    TagDic["VB"] = []
    for tok,tag in pos_tags:
        if tag in NNPTag and tok not in NNPTrash:
            TagDic["NNP"].append(tok)
        elif tag in NounTag and tok not in NNPTrash:
            TagDic["NOUN"].append(tok)
        elif tag in JJTag:
            TagDic["JJ"].append(tok)
        elif tag in VEBTag and tok not in VEBTrash:
            TagDic["VB"].append(tok)
    return TagDic


def NumberOfMatch(claimTag,senTag,type):
    matchNum = 0
    if type=="NNP":
        if len(claimTag[type]) !=0 and len(senTag[type]) !=0:
            for cnnp in claimTag[type]:
                for sccp in senTag[type]:
                    if cnnp == sccp:
                        matchNum += 1
                        break
    elif type == "VB":
        # senVbN = senTag[type]+senTag["NOUN"]
        # claVbN = claimTag[type]+claimTag["NOUN"]
        if len(claimTag[type]) !=0 and len(senTag[type]) !=0:
            for cn in claimTag[type]:
                for sc in senTag[type]:
                    maxwup = max_sim_v(cn,sc)
                    if maxwup > 0.8 :
                        matchNum += 1
                        break


    elif type == "NOUN":
        senNoun = senTag[type]+senTag["NNP"]
        if len(claimTag[type]) !=0 and len(senNoun) !=0:
            for cn in claimTag[type]:
                for sc in senNoun:
                    maxwup = max_sim_n(cn,sc)
                    if maxwup > 0.8 :
                        matchNum += 1
                        break

    else:
        if len(claimTag[type]) !=0 and len(senTag[type]) !=0:
            for cn in claimTag[type]:
                for sc in senTag[type]:
                    maxwup = max_sim_adj(cn,sc)
               
                    if maxwup > 0.8 :
                        matchNum += 1
                        break
    return matchNum



def max_sim_n(word1,word2):
    maxwup =0
    temp1 = 0
    temp2 = 0
    if (word1.isdigit() and word2.isdigit()) or (word1.isalnum() and word2.isalnum()):
        if word1 == word2:
            maxwup = 1.0
    else:
        for synset1 in wn.synsets(word1,"n"):
            for synset2 in wn.synsets(word2,"n"):
                temp1 = wn.synset(synset1.name()).wup_similarity(wn.synset(synset2.name()))
                if temp1 > maxwup:
                    maxwup = temp1
    return maxwup

def max_sim_v(word1,word2):
    maxwup_v =0
    temp1 = 0
    for synset1 in wn.synsets(word1,"v"):
        for synset2 in wn.synsets(word2,"v"):
            temp1 = wn.synset(synset1.name()).wup_similarity(wn.synset(synset2.name()))
            if temp1 > maxwup_v:
                maxwup_v = temp1
    return maxwup_v

def max_sim_adj(word1,word2):
    maxwup =0.0
    temp1 = 0
    temp2 = 0
    if word1 == word2:
        maxwup = 1.0
    else:
        for synset1 in wn.synsets(word1):
            for synset2 in wn.synsets(word2):
                temp1 = wn.synset(synset1.name()).wup_similarity(wn.synset(synset2.name()))
                if temp1 is not None:
                    if temp1 > maxwup:
                        maxwup = temp1
    return maxwup


#function for judge label algorithm
def judgeResult(claim,raw_results,Simi,numOfEvi,counDic):
    ans=""
    
    claimTag = getDifferentTag(claim)
    # for rs,id in raw_results:
    #     if id.isdigit():
    #         results.append((rs,int(id)))
    
    if len(raw_results) != 0:
        claimTag = getDifferentTag(claim)
        refSent = []
        claimNNP = len(claimTag["NNP"])
        claimNN = len(claimTag["NOUN"])
        claimJJ = len(claimTag["JJ"])
        claimVB = len(claimTag["VB"])
        for sent in raw_results:
            senTag = getDifferentTag(sent)
            MatNNP = NumberOfMatch(claimTag,senTag,"NNP")
            MatNOUN = NumberOfMatch(claimTag,senTag,"NOUN")
            MatJJ = NumberOfMatch(claimTag,senTag,"JJ")
            MatVB = NumberOfMatch(claimTag,senTag,"VB")
            if claimNNP>0:
                Simi[0] += MatNNP
                numOfEvi[0] += 1
                counDic["NNP"][MatNNP] += 1
            if claimNN>0:
                Simi[1] += MatNOUN
                numOfEvi[1] += 1
                counDic["NOUN"][MatNOUN] += 1
            if claimJJ>0:
                Simi[3] += MatJJ
                numOfEvi[3] += 1
                counDic["JJ"][MatJJ] += 1
            if claimVB>0:
                Simi[2] += MatVB
                numOfEvi[2] += 1
                counDic["VB"][MatVB] += 1
    return Simi,numOfEvi,counDic

            



def checkInEvi(raw_results,CorrectEvi):
    eviSent = []
    match = 0
    for rs in raw_results:
        rsTitle = rs["title"].replace(" ","_")
        rsSent = int(rs["sent"])
        for evi in CorrectEvi:
            if rsTitle == evi[0] and rsSent == evi[1]:
                match += 1
                eviSent.append(rs["content"])
    return match,eviSent

def checkInEvi2(results,CorrectEvi):
    match = 0
    for sent,number in results:
        rsTitle = sent.split(":")[0][:-1].replace(" ","_")
        rsSent = number
        for evi in CorrectEvi:
            if rsTitle == evi[0] and rsSent == evi[1]:
                match += 1
    return match


def QuerySentInSim(title,SentNum):
    # SetIndex("TestIndex")
    # use Fuzzy
    # parser.add_plugin(FuzzyTermPlugin())
    NewTitle = title.replace("_"," ")
    myquery = parser.parse(NewTitle)
    results =searcher.search(myquery,limit=None)
    acturalResult = ""
    for rs in results:
        if rs["title"] == NewTitle:
            if rs["sent"] == str(SentNum):
                allcon = rs['title']+" , "+rs['content']
                return allcon
    return ""



# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
ix = open_dir("WikiSplit")

# use BM25F to calculate the similarity between title and query
searcher = ix.searcher()
schema = ix.schema
parser = QueryParser("title",schema)
# parser.add_plugin(MultifieldPlugin(["title","sent"]))

# claimsWithId contains (id, claim) pairs
claimTogether,claimsWithId = readClaims("train.json")

# Get the entities from all claims(for time saving)
st = StanfordNERTagger('english.conll.4class.distsim.crf.ser.gz','stanford-ner.jar')
NerSen = st.tag(nltk.word_tokenize(claimTogether))

# NumOfClaim record the number of claims that have been processed
NumOfClaim = 0
TaggedClaim = []
claims = {}
# start = time.time()
#output json dict
# AvgNN = []
# AvgJJ = []
Simi = [0.0,0.0,0.0,0.0]
numOfEvi=[0,0,0,0]
counDic = defaultdict(Counter)

for TagToken in NerSen:
    # print(TagToken)
    if TagToken[0] != "SPLITATHERE":
        TaggedClaim.append(TagToken)

    else:
        # s = time.time()
        key = claimsWithId[NumOfClaim][0]
        claim = claimsWithId[NumOfClaim][1]
        # print("----------------------------------------------")
        # print(claim)
        CorrectEvi = claimsWithId[NumOfClaim][2]
        NumOfClaim += 1
        # print(NumOfClaim)
        # claims[key] = {"claim":claim}
        results = []
        for evidence in CorrectEvi:
            re = ""
            re = QuerySentInSim(evidence[0],evidence[1])
            if re != "":
                results.append(re)
        if len(results) != 0:
            Simi,numOfEvi,counDic = judgeResult(claim,results,Simi,numOfEvi,counDic)
        TaggedClaim = []
        print(NumOfClaim)

Simil = {"NNP":Simi[0]/numOfEvi[0],"NOUN":Simi[1]/numOfEvi[1],"VB":Simi[2]/numOfEvi[2],"JJ":Simi[3]/numOfEvi[3]}






with open('typeSimiSupports.json',"wb") as f:
    f.truncate()
    f.write((json.dumps(Simil,indent = 4).encode("utf-8")))
    f.write((json.dumps(counDic,indent = 4).encode("utf-8")))


# print(AvgNN)
# NounAvg = 0.0
# for avgs in AvgNN:
#     NounAvg += avgs
# if len(AvgNN) != 0:
#     NounAvg = NounAvg/len(AvgNN)

# print(AvgNN)
# JJAvg = 0.0
# for avgs in AvgJJ:
#     JJAvg += avgs
# if len(AvgJJ) != 0:
#     JJAvg = JJAvg/len(AvgJJ)

# print("NounAvg: ")
# print(NounAvg)
# print("JJAvg: ")
# print(JJAvg)
        



# end = time.time()
# print(end-start)
# print(NumOfClaim)
