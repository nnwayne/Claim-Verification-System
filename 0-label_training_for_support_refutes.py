from whoosh.fields import Schema, TEXT
import os.path
from whoosh.index import open_dir
from whoosh.index import create_in
from whoosh.query import *
from whoosh.qparser import QueryParser,FuzzyTermPlugin,SequencePlugin,MultifieldPlugin
import time,json,nltk
from nltk.tag.stanford import StanfordNERTagger
import nltk
from nltk.corpus import wordnet as wn
import json
import os
def SetIndex(indexName):
    schema = Schema(title=TEXT(stored=True), content=TEXT(stored=True), sent=TEXT(stored=True))
    if not os.path.exists(indexName):
        os.mkdir(indexName)
    ix = create_in(indexName, schema)
    ix = open_dir(indexName)
    filePath = "wiki-pages-text"
    pathDir=os.listdir(filePath)
    i=0
    for allDir in pathDir:
        writer = ix.writer()
        importDoc(writer,allDir)
        writer.commit()
        i += 1
        print(i)
    print("importDocFinish")


def importDoc(writer,allDir):
    child = "wiki-pages-text" + '/' + allDir
    fopen=open(child,'r')
    for lines in fopen.readlines():
        SLines = lines.split(" ")
        newTitle = SLines[0].replace("_"," ")
        linecontent = newTitle+":"+lines[len(SLines[0])+len(SLines[1])+2:]
        writer.add_document(title=SLines[0], content=linecontent, sent=SLines[1])


def getNER(sent):
    st = StanfordNERTagger('english.conll.4class.distsim.crf.ser.gz',
                           'stanford-ner.jar')
    NerSen = st.tag(sent.split())
    Entities = {}
    EntityName = "";
    length = len(NerSen)
    for i in range(length):
        if NerSen[i][1] != 'O':
            if i >0 and NerSen[i - 1][1] != 'O':
                EntityName += '_'
            EntityName += NerSen[i][0]
            if i == length-1 or (i < length-1 and NerSen[i + 1][1] == 'O'):
                Entities[EntityName] = NerSen[i][1]
                EntityName = ""
    words=nltk.word_tokenize(sent)
    pos_tags =nltk.pos_tag(words)
    EntityKey = []
    for key in Entities.keys(): EntityKey.append(key)
    for token,tag in pos_tags:
        if tag == "NNP":
            isContain = False
            for key in EntityKey:
                if token in key:
                    isContain = True
            if not isContain:
                Entities[token] = "NNP"
    return Entities

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

def tagSent(sent):
    words=nltk.word_tokenize(sent)
    pos_tags =nltk.pos_tag(words)
    EntityKey = []
    Entities = {}
    EntityName = ""
    length = len(pos_tags)
    NounTag = ["NN","NNS","CD"]
    for i in range(length):
        if pos_tags[i][1] == 'NNP':
            if i >0 and pos_tags[i - 1][1] == 'NNP':
                EntityName += '_'
            EntityName += pos_tags[i][0]
            if i == length-1 or (i < length-1 and pos_tags[i + 1][1] != 'NNP'):
                Entities[EntityName] = pos_tags[i][1]
                EntityName = ""
        if pos_tags[i][1] in NounTag:
            EntityName += pos_tags[i][0]
            j=i-1
            while j>=0 and pos_tags[j][1] == "JJ":
                EntityName = pos_tags[j][0] +" "+EntityName
                j -= 1
            Entities[EntityName] = "OTHER"
            EntityName=""

    st = StanfordNERTagger('english.conll.4class.distsim.crf.ser.gz',
                           'stanford-ner.jar')
    NerSen = st.tag(sent.split())
    for token,type in Entities.items():
        for word,tag in NerSen:
            if (word in token) and ("PERSON" == tag):
                Entities[token] = "PERSON"
    return Entities

def IsManyPerson(Entities):
    i =0
    for value in Entities.values():
        if value == "PERSON":
            i += 1
    if i>1:
        return True
    return False


def generateQuery(evidences):
    print(evidences)
    QueryEvi = []
    for evidence in evidences:
        QueryEvi.append(evidence[0])
        QueryEvi.append(evidence[1])
    return QueryEvi


def QuerySent(query,schema,SentNum):
    # SetIndex("TestIndex")
    # use Fuzzy
    parser =QueryParser(None,schema)
    parser.add_plugin(MultifieldPlugin(["title","sent"]))
    # parser.add_plugin(FuzzyTermPlugin())
    myquery = parser.parse(query)
    results =searcher.search(myquery)
    acturalResult = ""
    for rs in results:
        if rs["sent"] == str(SentNum):
            return rs["content"]


def readClaims(read_file):
    # count is used to control the number of claims that should be read
    claimTogether=""
    claimsWithId = []
    twolabel = ["SUPPORTS","REFUTES"]

    nm =0
    with open(read_file, "r") as r_file:
        trainDic = json.load(r_file)
        for key, value in trainDic.items():
            #if value["label"] == "SUPPORTS":
            nm += 1
            claimsWithId.append((key,value["claim"],value["evidence"],value["label"]))
            claimTogether = claimTogether + value["claim"] + " SPLITATHERE "
            if nm > 4999:
                break
    return claimTogether,claimsWithId


#function for judge the label for claims
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
    # to be modified

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

def tag_nnp(word_list): 
    ret = []
    word_list = nltk.word_tokenize(word_list)
    pos_tags =nltk.pos_tag(word_list)
    tags = set(['NNP','NNPS'])
    for word,pos in pos_tags:
        if (pos in tags):
            ret.append(word)
    return ret

def tag_v(word_list): 
    ret_v= []
    word_list = nltk.word_tokenize(word_list)
    pos_tags =nltk.pos_tag(word_list)
    tags = set(['VB','VBD','VBG','VBN','VBP','VBZ'])
    for word,pos in pos_tags:
        if (pos in tags):
            if (word != 'is' and word !='are') and (word !='was' and word != 'were'):
                ret_v.append(word)
    return ret_v
# function for get the max similarity value between word1 and word2
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

def max_sim_v(word1,word2):
    maxwup_v =0
    temp1 = 0
    
    for synset1 in wn.synsets(word1,"v"):
        for synset2 in wn.synsets(word2,"v"):
            temp1 = wn.synset(synset1.name()).wup_similarity(wn.synset(synset2.name()))
            if temp1 > maxwup_v:
                maxwup_v = temp1
    return maxwup_v

def max_sim_nnp(word1,word2):
    maxwup_nnp =0

    if word1.lower() == word2.lower():
        maxwup_nnp = 1
    return maxwup_nnp
def replace_sth(results):
    new_results = []
    for evi in results:
        if evi:
            evi1 = evi.replace('-LRB-', '(')
            evi2 =evi1.replace('-RRB-', ')')
            evi3 = evi2.replace('_', ' ')
            new_results.append(evi3)
        else:
            new_results.append("wrong")
    return new_results
                
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
 
def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None
 
    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None

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
def symmetric_sentence_similarity(sentence1, sentence2):
    """ compute the symmetric sentence similarity using Wordnet """
    return (sentence_similarity(sentence1, sentence2) + sentence_similarity(sentence2, sentence1)) / 2 

 
model = {}
max_info = {}
ix = open_dir("WikiSplit")

# use BM25F to calculate the similarity between title and query
searcher = ix.searcher()
schema = ix.schema
parser = QueryParser("title",schema)
# parser.add_plugin(MultifieldPlugin(["title","sent"]))

# claimsWithId contains (id, claim) pairs
claimTogether,claimsWithId = readClaims("train.json")

# Get the entities from all claims(for time saving)
st = StanfordNERTagger('english.conll.4class.distsim.crf.ser.gz',
                           'stanford-ner.jar')
NerSen = st.tag(nltk.word_tokenize(claimTogether))

# NumOfClaim record the number of claims that have been processed
NumOfClaim = 0
TaggedClaim = []
claims = {}
start = time.time()
#output json dict
# AvgNN = []
# AvgJJ = []

count = 0
for TagToken in NerSen:
    # print(TagToken)
    if TagToken[0] != "SPLITATHERE":
        TaggedClaim.append(TagToken)
        #print(1)

    else:
        # s = time.time()
        key = claimsWithId[NumOfClaim][0]
        claim = claimsWithId[NumOfClaim][1]
        label = claimsWithId[NumOfClaim][3]
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

        other = []
        max_set = {}
        max_set_v = {}
        max_set_nnp = {}
        max_info = {}
        sentence_sim = 0
        evidence = []
        count_ng = 0
        words_ret = tag(claim)
        full_sentence_similarity = []
        print(results)
        count+=1
        print(count)
        if len(results) != 0:

            results = replace_sth(results)
            for word1 in words_ret:
                max_num = 0
                for evi in results:
                    res_set = tag(evi)
                    #print(res_set)
                    for word2 in res_set:
                        if max_num < max_sim(word1,word2):
                            max_num = max_sim(word1,word2)
                max_set[word1] = max_num
                sentence_sim += max_num
     
            words_ret_v = tag_v(claim)
            

            for w in words_ret_v:
                max_num = 0
                
                for evi_v in results:
                    res_set = tag_v(evi_v)
                    for rw in res_set:
                        if max_num < max_sim_v(w,rw):
                            max_num = max_sim_v(w,rw)
                max_set_v[w] = max_num
                sentence_sim += max_num

            words_ret_nnp = tag_nnp(claim)
            
            for w in words_ret_nnp:
                max_num = 0
                for nnp in results:
                    res_set = tag_nnp(nnp)
                    for rw in res_set:
                        if max_num < max_sim_nnp(w,rw):
                            max_num = max_sim_nnp(w,rw)
                max_set_nnp[w] = max_num
                sentence_sim += max_num
            full_sentence_similarity = []
            for sentence in results:
                full_sentence_similarity.append(symmetric_sentence_similarity(claim, sentence))

        else:
            if label == 'NOT ENOUGH INFO':
                count_ng +=1
            else:
                other.append(label)
        max_info = max_info_(max_set,max_set_v,max_set_nnp)

        if (len(max_set)+len(max_set_v)+len(max_set_nnp)) !=0:
            sentence_sim = sentence_sim/(len(max_set)+len(max_set_v)+len(max_set_nnp))
        model[key] = {
            "claim": claim,
            "label": label,
            "max_set_n":max_set,
            "max_set_v":max_set_v,
            "max_set_nnp":max_set_nnp,
            "max_info":max_info,
            "count_ng":count_ng,
            "sentence_sim":sentence_sim,
            "full_sentence_similarity": full_sentence_similarity,
            "sth wrong":other
        }
        TaggedClaim = []

        
    
  
    
supports_max_n_avg_s = 0
supports_max_n_sum_s = 0
supports_max_v_avg_s = 0
supports_max_v_sum_s = 0
supports_max_nnp_avg_s = 0
supports_max_nnp_sum_s = 0
sentence_similarity_sum_s= 0
count_nnp1_n0_s= 0

supports_max_n_avg_r = 0
supports_max_n_sum_r = 0
supports_max_v_avg_r = 0
supports_max_v_sum_r = 0
supports_max_nnp_avg_r = 0
supports_max_nnp_sum_r = 0
sentence_similarity_sum_r= 0
count_nnp1_n0_r= 0

count_sup= 0
count_re = 0
count_nei = 0
count_not_r = 0
count_not_s = 0
count_not_ng = 0
count_only_r = 0
count_only_s = 0
count_only_ng = 0
count_0_n_s = 0
count_0_v_s = 0
count_0_nnp_s = 0
count_0_n_r = 0
count_0_v_r = 0
count_0_nnp_r = 0
count_0_n_nei = 0
count_0_v_nei = 0
count_0_nnp_nei = 0
full_sim_max_s = 0
full_sim_min_s = 0
full_sim_avg_s = 0
full_sim_max_r = 0
full_sim_min_r = 0
full_sim_avg_r = 0
full_sim_max_nei = 0
full_sim_min_nei = 0
full_sim_avg_nei = 0
for key in model:
    full_sim_max = 0
    full_sim_min = 1000
    full_sim_avg = 0
    full_sim_sum = 0
    if len(model[key]["full_sentence_similarity"])!=0:
        for sim in model[key]["full_sentence_similarity"]:
            full_sim_sum += sim
            if sim > full_sim_max:
                full_sim_max = sim
            if sim < full_sim_min:
                full_sim_min = sim
        full_sim_avg = full_sim_sum/len(model[key]["full_sentence_similarity"])
    if model[key]['label'] == "SUPPORTS":
        for value in nltk.word_tokenize(model[key]["claim"]):
            if value == 'not':
                count_not_s +=1
            if value == 'only':
                count_only_s +=1
        count_sup +=1
        supports_max_n_sum_s += model[key]["max_info"]["max_n"]
        supports_max_n_avg_s += model[key]["max_info"]["avg_n"]
        supports_max_v_sum_s += model[key]["max_info"]["max_v"]
        supports_max_v_avg_s += model[key]["max_info"]["avg_v"]
        supports_max_nnp_sum_s += model[key]["max_info"]["max_nnp"]
        supports_max_nnp_avg_s += model[key]["max_info"]["avg_nnp"]
        sentence_similarity_sum_s +=model[key]["sentence_sim"]
        count_0_n_s += model[key]['max_info']["count_0_n"]
        count_0_v_s += model[key]['max_info']['count_0_v']
        count_0_nnp_s += model[key]['max_info']['count_0_nnp']
        full_sim_max_s += full_sim_max
        full_sim_min_s += full_sim_min
        full_sim_avg_s += full_sim_avg
        if  model[key]["max_info"]["avg_nnp"] == 1 and len(model[key]["max_set_n"]) == 0:
            count_nnp1_n0_s +=1
    if model[key]['label'] == "REFUTES":
        for value in nltk.word_tokenize(model[key]["claim"]):
            if value == 'not':
                count_not_r +=1
            if value == 'only':
                count_only_r +=1
        count_re +=1
        supports_max_n_sum_r += model[key]["max_info"]["max_n"]
        supports_max_n_avg_r += model[key]["max_info"]["avg_n"]
        supports_max_v_sum_r += model[key]["max_info"]["max_v"]
        supports_max_v_avg_r += model[key]["max_info"]["avg_v"]
        supports_max_nnp_sum_r += model[key]["max_info"]["max_nnp"]
        supports_max_nnp_avg_r += model[key]["max_info"]["avg_nnp"]
        sentence_similarity_sum_r +=model[key]["sentence_sim"]
        count_0_n_r += model[key]['max_info']["count_0_n"]
        count_0_v_r += model[key]['max_info']['count_0_v']
        count_0_nnp_r += model[key]['max_info']['count_0_nnp']
        full_sim_max_r += full_sim_max
        full_sim_min_r += full_sim_min
        full_sim_avg_r += full_sim_avg
        if  model[key]["max_info"]["avg_nnp"] == 1 and len(model[key]["max_set_n"]) == 0:
            count_nnp1_n0_r +=1
    
    if model[key]['label'] == "NOT ENOUGH INFO":
        count_nei +=1
        count_0_n_nei += model[key]['max_info']["count_0_n"]
        count_0_v_nei += model[key]['max_info']['count_0_v']
        count_0_nnp_nei += model[key]['max_info']['count_0_nnp']
        full_sim_max_nei += full_sim_max
        full_sim_min_nei += full_sim_min
        full_sim_avg_nei += full_sim_avg
        for value in nltk.word_tokenize(model[key]["claim"]):
            if value == 'not':
                count_not_ng +=1
            if value == 'only':
                count_only_ng +=1 
    

with open('sen_sim_model_5k.json',"wb") as f:
    f.write((json.dumps(model,indent = 4).encode("utf-8")))
    f.close()
parameter = {}
if (count_sup !=0 and count_re !=0 )and count_nei !=0:
    parameter = {
        'supports_max_n_sum_s': supports_max_n_sum_s/count_sup,
        'supports_max_n_avg_s': supports_max_n_avg_s/count_sup,
        'supports_max_v_sum_s': supports_max_v_sum_s/count_sup,
        'supports_max_v_avg_s': supports_max_v_avg_s/count_sup,
        'supports_max_nnp_sum_s':supports_max_nnp_sum_s,
        'supports_max_nnp_avg_s': supports_max_nnp_avg_s,
        'count_not_s':count_not_s,
        'count_0_n_s':count_0_n_s,
        'count_0_v_s':count_0_v_s,
        'count_0_nnp_s':count_0_nnp_s,
        'full_sim_max_s':full_sim_max_s/count_sup,
        'full_sim_min_s' :full_sim_min_s/count_sup,
        'full_sim_avg_s' :full_sim_avg_s/count_sup,
        'count of avg(nnp) is 1 and no n in the claim for label is support': count_nnp1_n0_s,
        'sentence_similarity__sum_s':sentence_similarity_sum_s/count_sup,
        'total support label': count_sup,
        'refutes_max_n_sum_r': supports_max_n_sum_r/count_re,
        'refutes_max_n_avg_r': supports_max_n_avg_r/count_re,
        'refutes_max_v_sum_r': supports_max_v_sum_r/count_re,
        'refutes_max_v_avg_r': supports_max_v_avg_r/count_re,
        'full_sim_max_r':full_sim_max_r/count_re,
        'full_sim_min_r' :full_sim_min_r/count_re,
        'full_sim_avg_r' :full_sim_avg_r/count_re,
        'refutes_max_nnp_sum_r': supports_max_nnp_sum_r,
        'refutes_max_nnp_avg_r': supports_max_nnp_avg_r,
        'count_not_r':count_not_r,
        'count_0_n_r':count_0_n_r,
        'count_0_v_r':count_0_v_r,
        'count_0_nnp_r':count_0_nnp_r,
        'count_not_r':count_not_r,
        'count of avg(nnp) is 1 and no n in the claim for label is refuntes': count_nnp1_n0_r,
        'sentence_similarity_sum_r':sentence_similarity_sum_r/count_re,
        'count_not_ng':count_not_ng,
        'total refutes lable': count_re,
        'total not enough info label': count_ng,
        'count_0_n_nei':count_0_n_nei/count_nei,
        'count_0_v_nei':count_0_v_nei/count_nei,
        'count_0_nnp_nei':count_0_nnp_nei/count_nei,
        'full_sim_max_r':full_sim_max_nei/count_nei,
        'full_sim_min_r' :full_sim_min_nei/count_nei,
        'full_sim_avg_r' :full_sim_avg_nei/count_nei,
        'total not enough label need add this number': count_nei,
        'count_only_ng':count_only_ng,
        'count_only_s':count_only_s,
        'count_only_r':count_only_r

    }
with open('sen_sim_calculate_5k.json',"wb") as f1:
    f1.write((json.dumps(parameter,indent = 4).encode("utf-8")))
    f1.close()

end = time.time()
print(end-start)