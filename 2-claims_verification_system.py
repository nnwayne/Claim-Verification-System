from whoosh.fields import Schema, TEXT, STORED
import os.path,os,nltk,time,time,json,nltk,requests,gzip,spacy
from whoosh.index import open_dir
from whoosh.index import create_in
from whoosh.query import *
from whoosh.qparser import QueryParser,FuzzyTermPlugin,SequencePlugin
from nltk.tag.stanford import StanfordNERTagger
from whoosh import scoring
from pathlib import Path
from collections import Counter
from math import log, sqrt
from nltk.corpus import wordnet as wn


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


# These 5 commented funtions are used for Stanford NER

# def tagSent2(TaggedClaim):
#     # received tagged claim, and gather entities
#     Entities = []
#     length = len(TaggedClaim)
#     lst = iter(range(length-1,-1,-1))
#     for i in lst:
#         if TaggedClaim[i][1] != 'O':
#             EntityName = TaggedClaim[i][0]
#             j=i-1
#             while j>-1:
#                 if TaggedClaim[j][1] != 'O':
#                     EntityName = TaggedClaim[j][0]+" "+EntityName
#                     lst.__next__()
#                     j -= 1
#                 else:
#                     Entities.append(EntityName)
#                     EntityName = ""
#                     break
#             if EntityName != "":
#                 Entities.append(EntityName)
#                 EntityName = ""
#     return Entities


# def tagSent(sent,OriEntities):
#     # process claims using POS tag, and gather entities
#     words=nltk.word_tokenize(sent)
#     pos_tags =nltk.pos_tag(words)
#     Entities = {}
#     EntityName = ""
#     Others = ""
#     Noun = ""
#     length = len(pos_tags)
#     NounTag = ["NN","NNS","CD"]
#     NNPTag = ["NNP","NN","NNPS","PRP"]
#     Entities["NNP"]=[]
#     Entities["NOUN"]=[]
#     FinalEntities = []
#     lst = iter(range(length-2,-1,-1))
#     for i in lst:
#         EntityName = ''
#         if pos_tags[i][1] == 'NNP':
#             InEn = False
#             for entity in OriEntities:
#                 if pos_tags[i][0] in entity:
#                     InEn = True
#             if InEn == False:
#                 j=i-1
#                 EntityName = pos_tags[i][0]
#                 while j>=0 and (pos_tags[j][1] in NNPTag):
#                     EntityName = pos_tags[j][0]+" "+EntityName
#                     lst.__next__()
#                     j -= 1
#                 Entities["NNP"].append(EntityName)
#                 # print(Entities)
            
                    

#         elif pos_tags[i][1] in NounTag:
#             Noun = pos_tags[i][0]
#             j=i-1
#             while j>=0 and (pos_tags[j][1] == "JJ" or pos_tags[j][1] in NounTag):
#                 Noun = pos_tags[j][0]+" "+Noun
#                 lst.__next__()
#                 j -= 1
#             Entities["NOUN"].append(Noun)

#         elif Others == "":
#             Others += pos_tags[i][0]
#         else:
#             Others = pos_tags[i][0]+" "+Others
#     Entities["OTHER"] = Others
#     if len(Entities["NNP"]) >0:
#         for i in range(len(Entities["NNP"])-1,-1,-1):
#         # for ent in Entities["NNP"]:
#             FinalEntities.append(Entities["NNP"][i])
#     elif len(OriEntities) == 0 and len(Entities["NOUN"]) >0:
#         for i in range(len(Entities["NOUN"])-1,-1,-1):
#         # for ent in Entities["NNP"]:
#             FinalEntities.append(Entities["NOUN"][i])

#     return FinalEntities


# def checkLargeWord(Entities):

#     # these words have a large number of related titles
#     # we should remove these words at the first step
#     LargeWord = ['American', 'United States', 'English', 'England', 'America', 'Canadian', 
#     'British', 'French', 'Canada', 'California', 'German', 'Indian', 'France', 'Boston', 
#     'Chinese', 'London', 'Europe', 'Russia', 'Paris', 'Japanese', 'United Kingdom', 'New York', 
#     'Spain', 'Austria', 'Colombia', 'Florida', 'Los Angeles','China', 'U.S.','U.S' 'O. J. Simpson', 
#     'Brazil', 'UK', 'South Korea','Google', 'Star Wars', 'European', 'Ireland', 'Beautiful', 
#     'Russian', 'San Francisco', 'Hawaii', 'Belgium', 'Chicago', 'Charles', 'Britain', 
#     'Pennsylvania', 'film', 'Las Vegas', 'Italy', 'Italian', 'Indonesia', 'World War II', 'Liverpool', 
#     'Academy Awards', 'Australian', 'Hollywood', 'Constantine', 'Singapore', 'Renaissance','Indiana']

    
#         # if len(Entities) == 1 and Entities[0] in LargeWord:
#         #     # weighting
#     TempEntities = Entities
#     if len(Entities) > 0:
#         for ent in Entities:
#             if ent in LargeWord:
#                 TempEntities.remove(ent)
#             # if len(TempEntities) == 0:
#             #     # weighting
#     return TempEntities


# def generateQuery(claim,TaggedClaim):
#     query = ""
#     qnnp = ""
#     Entities = []

#     # Entities = ["EntityName","EntityName",....]
#     Entities = tagSent2(TaggedClaim)
#     Entities += tagSent(claim,Entities)
#     Entities = checkLargeWord(Entities)
#     if len(Entities) != 0:
#         for nnp in Entities:
#             query = query+ "("+nnp+")~ OR "
#     else:
#         Entities = tagSent(claim,Entities)
#         Entities = checkLargeWord(Entities)
#         if len(Entities) != 0:
#             for nnp in Entities:
#                 query = query+ "("+nnp+")~ OR "

#     if query == "":
#         query = generateQuerySecond(claim)
#     return query[:-4]


# def generateQuerySecond(claim):
#     query = ""

#     words=nltk.word_tokenize(claim)
#     pos_tags =nltk.pos_tag(words)

#     NOUN = ["NN","NNS","CD","NNP"]

#     for i in range(len(pos_tags)):
#         # if pos_tags[i][1] in NOUN:
#         if pos_tags[i][1] == "NNP":
#             query = query+ "("+pos_tags[i][0]+")～ OR "
#     if query == "":
#         for i in range(len(pos_tags)):
#         # if pos_tags[i][1] in NOUN:
#             if pos_tags[i][1] in NOUN:
#                 query = query+ "("+pos_tags[i][0]+")～ OR "
#     return query



def QuerySent(query):
    # only return the 100 most similar titles with theirs contents
    myquery = parser.parse(query)
    results =searcher.search(myquery,limit=30)
    return results


def readClaims(read_file):
    # count is used to control the number of claims that should be read
    claimsWithId = []
    with open(read_file, "r") as r_file:
        trainDic = json.load(r_file)
        for key, value in trainDic.items():
            claimsWithId.append((key,value["claim"]))
    return claimsWithId


def getDifferentTag(sent):
    words=nltk.word_tokenize(sent)
    pos_tags =nltk.pos_tag(words)
    NounTag = ["NN","NNS"]
    NNPTag = ["NNP","CD","NNPS"]
    JJTag = ["JJ","JJR","JJS","RB","RBR","RBS"]
    VEBTag = ["VB","VBG","VBD","VBN","VBP","VBZ"]
    TagDic={}
    TagDic["NNP"] = []
    TagDic["NOUN"] = []
    TagDic["JJ"] = []
    TagDic["VB"] = []
    for tok,tag in pos_tags:
        if tag in NNPTag:
            TagDic["NNP"].append(tok)
        elif tag in NounTag:
            TagDic["NOUN"].append(tok)
        elif tag in JJTag:
            TagDic["JJ"].append(tok)
        elif tag in JJTag:
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
        if len(claimTag[type]) !=0 and len(senTag[type]) !=0:
            for cn in claimTag[type]:
                for sc in senTag[type]:
                    maxwup = max_sim_v(cn,sc)
                    
                    if maxwup > 0.8 :
                        matchNum += 1
                        break

    elif type == "NOUN":
        if len(claimTag[type]) !=0 and len(senTag[type]) !=0:
            for cn in claimTag[type]:
                for sc in senTag[type]:
                    maxwup = max_sim_n(cn,sc)
                   
                    if maxwup > 0.8 :
                        matchNum += 1
                        break

    else:
        if len(claimTag[type]) !=0 and len(senTag[type]) !=0:
            for cn in claimTag[type]:
                for sc in senTag[type]:
                    maxwup = max_sim_n(cn,sc)
                    
                    if maxwup > 0.8 :
                        matchNum += 1
                        break
    return matchNum

#get the max similarity between two nouns among all synsets
def max_sim_n(word1,word2):
    maxwup =0
    temp1 = 0
    temp2 = 0
    if word1 == word2:
        maxwup = 1.0
    else:
        for synset1 in wn.synsets(word1,"n"):
            for synset2 in wn.synsets(word2,"n"):
                temp1 = wn.synset(synset1.name()).wup_similarity(wn.synset(synset2.name()))
                if temp1 > maxwup:
                    maxwup = temp1
    return maxwup

#get the max similarity between two verbs among all synsets
def max_sim_v(word1,word2):
    maxwup_v =0
    temp1 = 0
    
    for synset1 in wn.synsets(word1,"v"):
        for synset2 in wn.synsets(word2,"v"):
            temp1 = wn.synset(synset1.name()).wup_similarity(wn.synset(synset2.name()))
            if temp1 > maxwup_v:
                maxwup_v = temp1
    return maxwup_v


#get the max similarity between two ad among all synsets
def max_sim_ad(word1,word2):
    maxwup =0
    temp1 = 0
    temp2 = 0
    if word1 == word2:
        maxwup = 1.0
    else:
        for synset1 in wn.synsets(word1):
            for synset2 in wn.synsets(word2):
                temp1 = wn.synset(synset1.name()).wup_similarity(wn.synset(synset2.name()))
                if temp1 > maxwup:
                    maxwup = temp1
    return maxwup

#funtion to get the max information according to max similarity of nouns, verbs and proper nouns
def max_info_(max_set,max_set_v,max_set_nnp):
    avg_n = 0
    sum_n = 0
    max_n = 0
    avg_v = 0
    sum_v = 0
    max_v = 0 
    avg_nnp = 0
    sum_nnp = 0
    max_nnp = 0 
    max_info = {}
    count_0_n = 0
    count_0_v = 0
    count_0_nnp = 0

    if len(max_set) > 0:
        max_n = max_set[max(max_set)]
        for n in max_set:
            sum_n = sum_n + max_set[n]
            if max_set[n] == 0:
                count_0_n +=1
        avg_n = sum_n/len(max_set)
    else:
        max_n = -1
    if len(max_set_v) > 0:
        max_v = max_set_v[max(max_set_v)]
        for v in max_set_v:
            sum_v = sum_v + max_set_v[v]
            if max_set_v[v] == 0:
                count_0_v +=1
        avg_v = sum_v/len(max_set_v)
    else:
        max_v = -1
    if len(max_set_nnp) > 0:
        max_nnp = max_set_nnp[max(max_set_nnp)]
        for nnp in max_set_nnp:
            sum_nnp = sum_nnp + max_set_nnp[nnp]
            if max_set_nnp[nnp] == 0:
                count_0_nnp +=1
        avg_nnp = sum_nnp/len(max_set_nnp)
    else:
        max_v = -1
        
    max_info['avg_n'] = avg_n
    max_info['max_n'] = max_n
    max_info['avg_v'] = avg_v
    max_info['max_v'] = max_v
    max_info['max_nnp'] = max_nnp
    max_info['avg_nnp'] = avg_nnp
    max_info['count_0_n']  = count_0_n
    max_info['count_0_v'] = count_0_v
    max_info['count_0_nnp'] = count_0_nnp
    return max_info

#function for tag word in the wordlist and get a list of word with tag nn, nns and cd
def tag(word_list): 
    ret = []
    word_list = nltk.word_tokenize(word_list)
    pos_tags =nltk.pos_tag(word_list)
    tags = set(['NN','NNS','CD'])
    for word,pos in pos_tags:
        if (pos in tags):
            ret.append(word)
    return ret

#function for tag word in the wordlist and get a list of word with tag nnp and nnps
def tag_nnp(word_list): 
    ret = []
    word_list = nltk.word_tokenize(word_list)
    pos_tags =nltk.pos_tag(word_list)
    tags = set(['NNP','NNPS'])
    for word,pos in pos_tags:
        if (pos in tags):
            ret.append(word)
    return ret

#function for tag word in the wordlist and get a list of word with tag verbs
def tag_v(word_list): 
    ret_v= []
    word_list = nltk.word_tokenize(word_list)
    pos_tags =nltk.pos_tag(word_list)
    tags = set(['VB','VBD','VBG','VBN','VBP','VBZ'])
    for word,pos in pos_tags:
        if (pos in tags):
            #delete the verb be, which occurs too many times in the claims without practical meaning
            if (word != 'is' and word !='are') and (word !='was' and word != 'were'):
                ret_v.append(word)
    return ret_v

# function for get the max similarity value between word1 and word2 (nouns)
def max_sim(word1,word2):
    maxwup =0
    temp1 = 0
    temp2 = 0
    if word1 == word2:
        maxwup = 1.0
    else:
        for synset1 in wn.synsets(word1,"n"):
            for synset2 in wn.synsets(word2,"n"):
                temp1 = wn.synset(synset1.name()).wup_similarity(wn.synset(synset2.name()))
                if temp1 > maxwup:
                    maxwup = temp1
    return maxwup

# function for get the max similarity value between word1 and word2 (proper nouns)
def max_sim_nnp(word1,word2):
    maxwup_nnp =0
    if word1.lower() == word2.lower():
        maxwup_nnp = 1
    return maxwup_nnp 

# function for replace some special signs in wiki sentences
def replace_sth(results):
    new_results = []
    for evi in results:
        if evi:
            evi1 = evi[0].replace('-LRB-', '(')
            evi2 =evi1.replace('-RRB-', ')')
            evi3 = evi2.replace('_', ' ')
            new_results.append((evi3,evi[1]))
        else:
            new_results.append("wrong")
    return new_results

#transfer penn tag to wordnet tag, which will be used to calculate the similarity
def penn_to_wn(tag):

    if tag == 'NN':
        return 'n'
    if tag.startswith('V'):
        return 'v'
    if tag.startswith('J'):
        return 'a'
    if tag.startswith('R'):
        return 'r'
    return None

#change the word pos tag to synset tag
def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None
    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None

#get the sentence similarity using wup similarity
def sentence_similarity(sentence1, sentence2):
    """ compute the sentence similarity using Wordnet """
    # Tokenize and tag
    sentence1 = nltk.pos_tag(nltk.word_tokenize(sentence1))
    sentence2 = nltk.pos_tag(nltk.word_tokenize(sentence2))
 
    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
 
    # Filter out the Nones
    synsets1 = list(filter(None, synsets1)) 
    synsets2 = list(filter(None, synsets2)) 
 
    score, count = 0.0, 0
    score_list = []
    best_score = None
    for synset in synsets1:
        for ss in synsets2:
            score_list.append(synset.wup_similarity(ss))
        # Get the similarity value of the most similar word in the other sentence
        score_list = list(filter(None, score_list)) 
        if len(score_list):
            best_score = max(score_list)

        # Check that the similarity could have been computed
        if best_score is not None:
            score += best_score
            count += 1
 
    # Average the values
    if count != 0:
        score /= count
    return score

#consider all the results as whole, calculte the data according to max set, which will be used to judge the label
def max_set_info(raw_results,claim):
    max_set = {}
    max_set_v = {}
    max_set_nnp = {}
    sentence_sim = 0
    results = []
    for rs in raw_results:
        if rs['sent'].isdigit():
            allcon = rs['title']+" , "+rs['content']
            results.append((allcon.replace("-"," - "),int(rs['sent'])))
    results = replace_sth(results)
    words_ret = tag(claim)
    for word1 in words_ret:
        max_num = 0
        for evi in results:
            res_set = tag(evi[0])
            for word2 in res_set:
                if max_num < max_sim(word1,word2):
                    max_num = max_sim(word1,word2)
        max_set[word1] = max_num
        sentence_sim += max_num 
                                   
    words_ret_v = tag_v(claim)
    for w in words_ret_v:
        max_num = 0
        for evi_v in results:
            res_set = tag_v(evi_v[0])
            for rw in res_set:
                if max_num < max_sim_v(w,rw):
                    max_num = max_sim_v(w,rw)
        max_set_v[w] = max_num
        sentence_sim += max_num                             

    words_ret_nnp = tag_nnp(claim)
    for w in words_ret_nnp:
        max_num = 0
        for nnp in results:
            res_set = tag_nnp(nnp[0])
            for rw in res_set:
                if max_num < max_sim_nnp(w,rw):
                    max_num = max_sim_nnp(w,rw)
        max_set_nnp[w] = max_num
        sentence_sim += max_num
                                   
    if (len(max_set)+len(max_set_v)+len(max_set_nnp)) !=0:
        sentence_sim = sentence_sim/(len(max_set)+len(max_set_v)+len(max_set_nnp))
    return max_set,max_set_v,max_set_nnp,sentence_sim

#classify the label, according the comoarison between the max_set_info that we get before 
#and the data that we get from training process
def judge_label(max_info,max_set,max_set_v,max_set_nnp,claim,results):
    label = ''
    not_only_judge = False

    for value in nltk.word_tokenize(claim):
            if value.lower() == 'not':
                not_only_judge = True
            if value.lower() == 'only':
                not_only_judge = True
            if value.lower() == 'no':
                not_only_judge = True
            if value.lower() == 'none':
                not_only_judge = True
            if value.lower() == 'never':
                not_only_judge = True
    if len(results) == 0:
        label = 'NOT ENOUGH INFO'
    elif not_only_judge == True:
        label = 'REFUTES'
    else:
        if max_info["count_0_nnp"] == len(max_set_nnp):
            label = 'NOT ENOUGH INFO'
        elif len(max_set_nnp) > 3 and max_info["count_0_nnp"] >= 2:
            label = 'NOT ENOUGH INFO'
        elif sentence_sim < 0.1:
            label = 'NOT ENOUGH INFO'
        else:
            if max_info["avg_nnp"] == 1:
                if len(max_set) == 0:
                    label = 'SUPPORTS'
                elif len(max_set) == max_info["count_0_n"]:
                    label = 'NOT ENOUGH INFO'
                elif max_info["avg_n"] > 0.5:
                    label = 'SUPPORTS'
                else:
                    label = 'REFUTES'
                                    
            elif max_info["avg_nnp"] < 0.4:
                label = 'NOT ENOUGH INFO'
            else:
                if sentence_sim > 0.6:
                    label = 'SUPPORTS'
                else:
                    label = 'REFUTES'
                    
    return label
 
 #when there are inconsistency between evidence and label judgement, for example, if there no evidence when the 
 #label is support or refutes, we use this function to generate evidence again
def generate_evidence_again(claim,raw_results):
    evidence = []
    temp_evidence = {}
    number_count = 0
    results = []
    for rs in raw_results:
        if rs['sent'].isdigit():
            allcon = rs['title']+" , "+rs['content']
            results.append((allcon.replace("-"," - "),int(rs['sent'])))
    
    if len(results) > 0:
        for pair in results:
            temp = 0
            title = ''
            title = pair[0].split(",")[0].replace(" - ","-").replace(" ","_")
            temp = sentence_similarity(claim,pair[0])
            temp_evidence[(title[:-1],int(pair[1]))] = temp
        temp_evidence = dict(sorted(temp_evidence.items(), key=lambda d: d[1],reverse=True))
        number_count = 0
        for key in temp_evidence:
            number_count += 1
            evidence.append(key)
            if number_count > 1:
                return evidence
                break
    else:
        evidence = []           
    return evidence

#function for judge evidence algorithm
def judgeResult(claim,raw_results):
    ans=""
    SUPPORTS = False
    REFUTES = False
    supEvi = []
    refEvi = []
    results = []
    for rs in raw_results:
        if rs['sent'].isdigit():
            allcon = rs['title']+" , "+rs['content']
            results.append((allcon.replace("-"," - "),int(rs['sent'])))
    
    if len(results) == 0:
        return []
    else:
        claimTag = getDifferentTag(claim)
        claimNNP = len(claimTag["NNP"])
        claimNN = len(claimTag["NOUN"])
        claimJJ = len(claimTag["JJ"])
        claimVB = len(claimTag["VB"])
        for sent,number in results:
            senTag = getDifferentTag(sent)
            MatNNP = NumberOfMatch(claimTag,senTag,"NNP")
            MatNN = NumberOfMatch(claimTag,senTag,"NOUN")
            MatJJ = NumberOfMatch(claimTag,senTag,"JJ")
            MatVB = NumberOfMatch(claimTag,senTag,"VB")
            title = sent.split(",")[0][:-1].replace(" - ","-").replace(" ","_")

            if MatNNP == claimNNP:
                if MatNN == claimNN:
                    if MatJJ == claimJJ:
                        SUPPORTS = True
                        supEvi.append([title,number])
                    else: 
                        REFUTES = True
                        refEvi.append([title,number])
                    if claimVB != MatVB:
                        return []
                else:
                    if claimVB == MatVB:
                        refEvi.append([title,number])
                        REFUTES = True
                        

            elif MatNNP != claimNNP:
                if MatNN == claimNN:
                    REFUTES = True
                    refEvi.append([title,number])
                # else:
                #     return "NOT ENOUGH INFO",[]


        if len(refEvi) == 0 and len(supEvi) == 0:
            return []

        toke_claim = nltk.word_tokenize(claim)
        DenyWord = ["not","only","no","never"]
        for tok in toke_claim:
            if tok in DenyWord:
                return supEvi

        if SUPPORTS == True:
            return supEvi
        elif REFUTES == True:
            return refEvi
        else:
            return []


def getNLP(sent):
    doc=nlp(sent)
    object = ""
    SubjectList = []
    ObjectList = []
    RootList = []
    AmodList = []
    subject=""
    getSubject=False
    getRoot = False
    skipNext = False
    for i in range(len(doc)):
        tok = doc[i]
        if skipNext:
            skipNext=False
        else:
            if getSubject:
                if str(tok) == "(":
                    j=i
                    subject += " "
                    while j<len(doc)-1 and ")" not in str(doc[j]) :
                        subject += str(doc[j])+" "
                        j += 1
                    subject += str(doc[j])
                    subject = subject.replace("( ","-LRB-").replace(" )","-RRB-").replace(")","-RRB-")
                elif tok.dep_ == "prep" and not getRoot:
                    j=i
                    subject += " "
                    while j<len(doc) and doc[j].dep_ != "ROOT":
                        subject += str(doc[j])+" "
                        j += 1
                    subject = subject[:-1]
                break

            if tok.dep_ != "nsubj" and tok.dep_ != "nsubjpass":
                subject += str(tok)+" "
            else:
                subject += str(tok)
                getSubject = True
            if tok.dep_ == "pobj" or tok.dep_ == "dobj" :
                subject = ""
                if doc[i].dep_ == "punct":
                    skipNext = True
            if tok.dep_ == "ROOT":
                getRoot = True

    if not getSubject:
        subject=""
        getRoot = False
        for i in range(len(doc)):
            tok = doc[i]
            if getSubject:
                if str(tok) == "(":
                    j=i
                    subject += " "
                    while j<len(doc)-1 and ")" not in str(doc[j]):
                        subject += str(doc[j])+" "
                        j += 1
                    subject += str(doc[j])
                    subject = subject.replace("( ","-LRB-").replace(" )","-RRB-").replace(")","-RRB-")
                elif tok.dep_ == "prep" and not getRoot:
                    j=i
                    subject += " "
                    while j<len(doc) and doc[j].dep_ != "ROOT":
                        subject += str(doc[j])+" "
                        j += 1
                    subject = subject[:-1]
                break
            # print(tok.dep_)
            if tok.dep_ != "dobj" and tok.dep_ != "pobj" :
                subject += str(tok)+" "
            else:
                subject += str(tok)
                getSubject = True
            if tok.dep_ == "ROOT":
                getRoot = True

    if subject[:3] == "The":
        subject = subject[4:]
    subject = subject.replace(" - ","-").replace(" '","'").replace(" :","-COLON-")
    if "'s" in subject:
        position = subject.find("'")
        SubjectList.append(subject[:position])
        SubjectList.append(subject[position+3:])
    else:
        SubjectList.append(subject)

    OBJTag = ["dobj","pobj","attr","conj"]
    OBJTrash = ["it","itself","they","them","themselves","date"]
    MODTag = ["amod","advmod","nummod"]
    NoVerb = ["bear","die","be"]
    lst = iter(range(len(doc)-1,-1,-1))
    lemtok=""
    for i in lst:
        if doc[i].dep_ in OBJTag and doc[i].dep_ not in OBJTrash:
            ObjectList.append(str(doc[i]))
            j=i-1
            while j>-1 and (doc[j].dep_ == 'compound' or doc[j].dep_ in OBJTag):
                if doc[i].dep_ not in OBJTag:
                    ObjectList.append(str(doc[j]))
                lst.__next__()
                j -= 1

        elif doc[i].dep_ in MODTag:
            AmodList.append(str(doc[i]))

    NLPClaim = {"Subject":SubjectList,"Object":ObjectList,"Verb":RootList,"Adj":AmodList}
    return NLPClaim


# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

ix = open_dir("WikiSplit")

# use BM25F to calculate the similarity between title and query
searcher = ix.searcher(weighting=scoring.BM25F)
schema = ix.schema
parser =QueryParser("title",schema)


# claimsWithId contains (id, claim) pairs
claimsWithId = readClaims("test-unlabelled.json")

# Get the entities from all claims(for time saving)
# st = StanfordNERTagger('english.conll.4class.distsim.crf.ser.gz',
#                            'stanford-ner.jar')
# NerSen = st.tag(nltk.word_tokenize(claimTogether))
# parser.add_plugin(FuzzyTermPlugin())

nlp = spacy.load("en")
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()


# NumOfClaim record the number of claims that have been processed
NumOfClaim = 0
claims = {}
start = time.time()

#output json dict
model = {}

#for label and evidence
max_set = {}
max_set_v = {}
max_set_nnp = {}
max_info = {}
sentence_sim = 0
evidence = []
temp_evi = []
for key,claim in claimsWithId:

    NumOfClaim += 1
    print("----------------------------------------------")
    print(claim)
    print(NumOfClaim)
    NLPClaim = getNLP(claim)

    query = NLPClaim["Subject"][0]

    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>LargeWord")
    # print(query)
    raw_results = QuerySent(query)

    # process raw_results to get 10 most similar Wiki sentences
    # results contains (sent, number) pairs
    # results = processResult(claim,raw_results)
    max_set,max_set_v,max_set_nnp,sentence_sim = max_set_info(raw_results,claim)
    max_info = max_info_(max_set,max_set_v,max_set_nnp)
    temp_evi = judgeResult(claim,raw_results)
    label = judge_label(max_info,max_set,max_set_v,max_set_nnp,claim,raw_results)
    
    #keep the consistency between label and evidence, for example, if the label is not enough info,
    # the evidence should be empty, otherwise, the evidence should not be empty
    if label == 'NOT ENOUGH INFO':
        evidence = []
    else:
        if temp_evi == []:
            evidence = generate_evidence_again(claim,raw_results)
        # we just keep the length of the evidence smaller than or equal to 5.
        elif len(temp_evi) <=5:
            evidence = temp_evi
        else:
            evidence = new[0:5]
    
    else:
        if evidence == []:
            evidence = generate_evidence_again(claim,raw_results)

    #store the dictionary with claim, label that we classify and evidence that we generate.
    claims[key] = {
        "claim": claim,
        "label": label,
        "evidence":evidence
    }

#save the results as a json file
with open('testoutput.json',"wb") as f:
    f.truncate()
    f.write((json.dumps(claims,indent = 4).encode("utf-8")))
        

