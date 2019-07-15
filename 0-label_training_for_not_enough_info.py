from whoosh.fields import Schema, TEXT, STORED
import os.path,nltk,time,time,json,nltk,requests,gzip
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
from nltk import pos_tag,word_tokenize

def SetIndex(indexName):
    schema = Schema(title=TEXT(stored=True), content=STORED, sent=STORED)
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
        linecontent = lines[len(SLines[0])+len(SLines[1])+2:]
        writer.add_document(title=newTitle, content=linecontent, sent=SLines[1])





def tagSent2(TaggedClaim):
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





def tagSent(sent):
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
            EntityName = pos_tags[i][0]
            j=i-1
            while j>=0 and (pos_tags[j][1] in NNPTag):
                EntityName = pos_tags[j][0]+" "+EntityName
                lst.__next__()
                j -= 1
            Entities["NNP"].append(EntityName)

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


def ConnectNouns(Nouns):
    qnoun = ""
    for noun in Nouns:
        q = noun.replace(" "," OR ")
        qnoun = q+" OR "+qnoun
    # qnoun="("+qnoun+")~ "
    qnoun = qnoun[:-4]
    return qnoun


def checkLargeWord(Entities):
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


def generateQuery5(claim,TaggedClaim):
    # print("claim")
    # print(claim)
    query = ""
    qnnp = ""

    # Entities = ["EntityName","EntityName",....]
    Entities = tagSent2(TaggedClaim)
    Entities = checkLargeWord(Entities)
    if len(Entities) != 0:
        for nnp in Entities:
            query = query+ "("+nnp+") OR "

    else:
        Entities = tagSent(claim)
        Entities = checkLargeWord(Entities)
        if len(Entities) != 0:
            for nnp in Entities:
                query = query+ "("+nnp+") OR "

    return query


def generateQueryLast(claim):
    query = ""

    words=nltk.word_tokenize(claim)
    pos_tags =nltk.pos_tag(words)

    # NOUN = ["NN","NNS","CD","NNP"]

    for i in range(len(pos_tags)):
        # if pos_tags[i][1] in NOUN:
        if pos_tags[i][1] == "NNP":
            query = query+ "("+pos_tags[i][0]+") OR "
    return query



def QuerySent(query):
    # SetIndex("TestIndex")
    # print(query)
    # use Fuzzy
    # parser.add_plugin(FuzzyTermPlugin())
    # parser.add_plugin(MultifieldPlugin(["content","title"]))
    myquery = parser.parse(query)
    results =searcher.search(myquery,limit=100)
    # results =searcher.search(myquery,limit=None)
    # print(results)
    # for rs in results:
    # print(rs["title"])
    return results


def readClaims(read_file):
    claimTogether=""
    cn = 0
    with open(read_file, "r") as r_file:
        trainDic = json.load(r_file)
        for key, value in trainDic.items():
            if value["label"] == "NOT ENOUGH INFO":
                claimsWithId.append([key,value["claim"]])
                claimTogether = claimTogether + value["claim"] + " SPLITATHERE "
                cn += 1
                if cn >1000:
                    break
    return claimTogether




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
                s = log(1+fdt)*log(N/ft)+s
        s = s/sqrt(index.doc_len[i])
        scores[i] += s

    return scores.most_common(k)


def processResult(claim,raw_results):
    raw_docs = []
    for rs in raw_results:
        allcon = rs['title']+" : "+rs['content']
        raw_docs.append(allcon)

    # processed_docs stores the list of processed docs
    processed_docs = []
    # vocab contains (term, term id) pairs
    vocab = {}
    # total_tokens stores the total number of tokens
    # total_tokens = 0

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
    processed_results = query_tfidf(stemmed_query, invindex, vocab)
    Top10Res = []
    for rank, res in enumerate(processed_results):
    #     # e.g RANK 1 DOCID 176 SCORE 0.426 CONTENT South Korea rose 1% in February from a year earlier, the
    #     print("RANK {:2d} DOCID {:8d} SCORE {:.3f} CONTENT {:}".format(rank+1,res[0],res[1],raw_docs[res[0]]))
        Top10Res.append(raw_docs[res[0]])
    return Top10Res




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
            if max_set_nnp[nnp] ==0:
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
            ret_v.append(word)
    return ret_v
# function for get the max similarity value between word1 and word2
def max_sim(word1,word2):
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

def max_sim_nnp(word1,word2):
    maxwup_nnp =0

    if word1 == word2:
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
              

# SetIndex("WikiMix")
# start = time.time()

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

ix = open_dir("WikiSplit")
searcher = ix.searcher(weighting=scoring.BM25F)
schema = ix.schema
parser =QueryParser("title",schema)

claimsWithId = []
claimTogether = readClaims("train.json")

st = StanfordNERTagger('english.conll.4class.distsim.crf.ser.gz',
                           'stanford-ner.jar')
NerSen = st.tag(nltk.word_tokenize(claimTogether))
NumOfClaim = 0
TaggedClaim = []
claims = {}
model = {}
max_info = {}
count_no_res = 0
full_sentence_similarity = []
for TagToken in NerSen:
    if TagToken[0] != "SPLITATHERE":
        TaggedClaim.append(TagToken)

    else:
        s = time.time()
        key = claimsWithId[NumOfClaim][0]
        claim = claimsWithId[NumOfClaim][1]
        NumOfClaim += 1
        claims[key] = {"claim":claim}
        print("----------------------------------------------")
        print(claim)
        query = generateQuery5(claim,TaggedClaim)
        if query == "":
            query = generateQueryLast(claim)
        raw_results = QuerySent(query)
        results = processResult(claim,raw_results)

        TaggedClaim = []
        words_ret = tag(claim)
        max_set = {}
        max_set_v = {}
        max_set_nnp = {}
        data = {}
        max_title = ''
        max_no = 0
        sentence_sim = 0
        print(results)
        if len(results)>0:
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
                    for rw in res_set:
                        if max_num < max_sim_nnp(w,rw):
                            max_num = max_sim_nnp(w,rw)
                max_set_nnp[w] = max_num
                sentence_sim += max_num
            for sentence in results:
                full_sentence_similarity.append(symmetric_sentence_similarity(claim, sentence))

        else:
            count_no_res +=1
        max_info = max_info_(max_set,max_set_v,max_set_nnp)
        if (len(max_set)+len(max_set_v)+len(max_set_nnp)) !=0:
            sentence_sim = sentence_sim/(len(max_set)+len(max_set_v)+len(max_set_nnp))
        model[key] = {
            "claim": claim,
            "max_set_n":max_set,
            "max_set_v":max_set_v,
            "max_set_nnp":max_set_nnp,
            "max_info":max_info,
            "sentence_sim":sentence_sim,
            "full_sentence_similarity": full_sentence_similarity
        }
        e = time.time()
        print(e-s)

        if NumOfClaim>5 :
            break

count_nei = 0
sentence_sim_nei = 0
count_0_n = 0
count_0_v = 0
count_0_nnp = 0
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
    
    count_0_n += model[key]['max_info']["count_0_n"]
    count_0_v += model[key]['max_info']['count_0_v']
    count_0_nnp += model[key]['max_info']['count_0_nnp']
    sentence_sim_nei += model[key]['sentence_sim']
    count_nei +=1

with open('trainmodelNotEnough.json',"wb") as f:
    f.write((json.dumps(model,indent = 4).encode("utf-8")))
    f.close()
parameter = {}
if count_nei != 0:
    parameter = {
        "sentence_sim_nei": sentence_sim_nei/count_nei,
        'total not enough info label': count_no_res,
        'total not enough label need add this number': count_nei,
        'count_0_n':count_0_n/count_nei,
        'count_0_v':count_0_v/count_nei,
        'count_0_nnp':count_0_nnp/count_nei
    }
with open('calculateNotEnough.json',"wb") as f1:
    f1.write((json.dumps(parameter,indent = 4).encode("utf-8")))
    f1.close()

