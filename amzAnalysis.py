#!/usr/local/bin/python3
import re
import datetime
import numpy as np
import pandas as pd
import nltk
#from IPython.display import display
import io
import collections
import os
import matplotlib.pyplot as plt
from nltk.tag import StanfordPOSTagger
from nltk.parse.stanford import StanfordDependencyParser
from gensim.models.word2vec import Word2Vec
from sklearn.svm import SVC
import json
#%matplotlib inline

#os.environ["JAVA_HOME"] = "/Library/Java/JavaVirtualMachines/jdk1.8.0_25.jdk/Contents/Home" 
#os.environ["CLASSPATH"] = "/home/chweng/Desktop/SemanticProj/StanfordNLP/jars"
#os.environ["STANFORD_MODELS"] = "/home/chweng/Desktop/SemanticProj/StanfordNLP/models"
#os.environ["CLASSPATH"] ='/Users/chweng/Desktop/StanfordNLP/jars'
#os.environ["STANFORD_MODELS"] ='/Users/chweng/Desktop/StanfordNLP/models'
os.environ["CLASSPATH"] ='/users/raemoen/Desktop/StanfordNLP/jars'
os.environ["STANFORD_MODELS"] ='/users/raemoen/Desktop/StanfordNLP/models'
def load_csv(date):
    '''載入資料庫抓下來的csv檔，將之輸出為pandas dataframe。
    '''
    df=pd.DataFrame.from_csv("data/reviews-%s.csv"%(date), encoding="utf-8",index_col=None)
    #.reset_index(drop=True)
    return df

def summary(df):
    '''確認資料數。
    '''
    prodTypes=["central","canister","handheld","robotic","stick","upright","wetdry"]
    #choose type
    for prodType in prodTypes:
        print(prodType,len(df.loc[df["ptype"]==prodType]["review"]))
    print("total reviews=",len(df),"\n")
    
def store_prodRevs_in_a_list(prodType,df):
    '''將pandas dataframe內的評論提取，存成一個List。該List除了含有評論，也含有該評論的索引，以方便查找。
    '''
    df=df.loc[df["ptype"]==prodType]
    pReviewsGroup=df.groupby(['ptype','pid'])

    reviews_list=[]
    for key,pReviews in pReviewsGroup:
        rids=pReviews[["rid","review"]].values[:,0].tolist()
        reviews=pReviews[["rid","review"]].values[:,1].tolist()

        ptypes=[key[0]]*len(rids)
        pids=[key[1]]*len(rids)
        reviews_list+=list(zip(ptypes,pids,rids,reviews))

    print("number of products=",len(pReviewsGroup))
    print("number of reviews in type %s=%i"%(prodType,len(reviews_list)))
    
    return reviews_list
    
def cleanText(review):
    '''清理各種特殊表情及符號。
    '''
    #print(len(review))
    review=review.lower().replace('\n','').replace(';',',').replace('!',',').replace('_','-') \
                 .replace('i.e.','').replace("doesn't","does not").replace("don't","do not") \
                 .replace("didn't","did not").replace("it's","it is") \
                 .replace("isn't","is not").replace("there's","there is").strip()
    review=re.sub("\.+,", ",",review)     # replace ......., with ,
    review=re.sub("\.+",".", review)      # replace ........ with .
    review=re.sub("\,+",",", review)      # replace ,,,,,,,, with ,
    review=re.sub('[\"\*]','', review)    # remove " and *
    
    #remove emoji
    emoji_pattern = re.compile("["  
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    review=re.sub(emoji_pattern,"", review)
    review=re.sub('(?::|;|=)(?:-)?(?:\)|\(|D|P)',"",review)
    # notice that, the question mark makes the preceding token
    # in the regular expression optional.
    # also, (?: xxx) is a non-capturing group
    # let us probably come back to this re later.
    return review
    
def splitSentences(review):
    '''斷句。
    '''
    rev=re.split(r' *[\.\?!][\'"\)\]]* *', review)
    return rev
    
def reviewsProcessing(reviews_list):
    '''將reviews_list內的評論，斷句後，轉換成含有該評論所有句子的一個清單。
    '''
    reviews_list=[(review[0],review[1],review[2],splitSentences(cleanText(review[3])))
                                   for review in reviews_list if review[3]==review[3]]
    return reviews_list
    
def extractSentencesOfTheSelectedReviews(prodIDs,reviews_list):
    # extract the sentences of the targeted products (those products are contained in the list: prodIDs)
    buf = io.StringIO(prodIDs)
    prodIDs=buf.readlines()
    prodIDs=list(map(lambda x:re.sub('[\n]','',x) ,prodIDs))
    
    selectedReviews=[review for review in reviews_list if(review[1] in prodIDs)]
    sentences=[(sentences[0],sentences[1],sentences[2],sentence) for sentences in selectedReviews for sentence in sentences[3]]
    print("number of sentences from the selected prodIDs=%i"%(len(sentences)))
    return sentences
    
def findMatchedSentences(pattern,sentences):
    sentsMatched=[sentence for sentence in sentences if pattern[0] in sentence[3]]
    print("number of the sentences matched the pattern '%s'=%i"%(pattern,len(sentsMatched)))
    return sentsMatched

def stanfordP(sentence):
    tagger = StanfordPOSTagger(model_filename=r'english-bidirectional-distsim.tagger')
    result=tagger.tag(sentence.split())
    return result

def stanfordDP(sentence,displayTree=0,showTriples=0):
    '''Stanford依存語法解析。若需印出依存圖則設定displayTree=1。
    '''
    #print(repr(sentence),'\n')
    parser = StanfordDependencyParser()
    res = list(parser.parse(sentence.split()))
    
    #print(res[0].tree(),'\n')
    #print(*res[0].tree(),'\n')

    rels=[rel for rel in res[0].triples()]
    if(showTriples!=0):
        for row in res[0].triples():
            print(row)
    
    if(displayTree!=0):
        for row in res[0].tree():
            #print(row)
            if type(row) is not str:
                #row.draw()
                display(row)
    return rels
    
def depRelWordsFind(targetWord,depPairs):
    '''和目標詞targetWord相鄰節點之字詞(dependent words)將存於清單depWords。
       另外，修飾目標詞的一些詞以及修飾那些詞的副詞，將存於清單relWords。於清單relWords中的字詞只有少數(一兩個字)，其中很可能包含重要的情緒字詞。
       而，於清單depWords儲存的詞則包含較多資訊，是鄰近targetWord的片段，應包含targetWord的語意。'''
    print('our target word= ',targetWord)
    # given a target word, i.e. a word (say, 'noise'), we find all the dependent words of that word

    targetWordType=targetWord[1]
    targetWord=targetWord[0]
    depWords=[targetWord,]
    #depWords=[]
    if(targetWordType=='JJ'):
        relWords=[(targetWord,targetWordType),]
    else:
        relWords=[]
    relations=['amod','compound','acl:relcl','nsubj','dobj']
    
    '''選依存樹內和目標詞相鄰的節點儲存於depWords。
       另外，選和目標詞以amod，compound等關係相連結的詞彙，儲存於relWords。'''
    for depPair in depPairs:
#         depPair0Type=nltk.pos_tag([depPair[0]])[0][1]
#         depPair1Type=nltk.pos_tag([depPair[1]])[0][1]
        # e.g. depPair= ('level', 'noise', 'compound')
        #('works', 'sunction', 'nsubj') ['amod', 'compound', 'acl:relcl', 'nsubj', 'dobj']
        if(targetWord in depPair[0]):
            depWords.append(depPair[1])
            if(depPair[2] in relations):
                relWords.append((depPair[1],depPair[2]))
        if(targetWord in depPair[1]):
            depWords.append(depPair[0])
            if(depPair[2] in relations):
                relWords.append((depPair[0],depPair[2]))

    '''標註depWords內的詞為負向，若它與其他詞彙有負向關係且它自己本身是動詞或形容詞。
       也就是說，我們不考慮將負向標籤加至名詞身上。'''

    for depPair in depPairs:
        depPair0Type=nltk.pos_tag([depPair[0]])[0][1]
        depPair1Type=nltk.pos_tag([depPair[1]])[0][1]
        if(depPair[0] in depWords) and (depPair[2] == 'neg') and ('JJ' in depPair0Type or 'VB' in depPair0Type or 'NN' in depPair0Type) and (depPair[0]!='not'):
            relWords.append(('not_'+depPair[0],depPair[2]))
            depWords.append('not')
        if(depPair[1] in depWords) and (depPair[2] == 'neg') and ('JJ' in depPair1Type or 'VB' in depPair1Type or 'NN' in depPair1Type) and (depPair[1]!='not'):
            relWords.append(('not_'+depPair[1],depPair[2]))
            depWords.append('not')

        if(depPair[0] in depWords) and (depPair[2] == 'neg') and (depPair0Type=='NN'):
            depWords.append('not')
        if(depPair[1] in depWords) and (depPair[2] == 'neg') and (depPair1Type=='NN'):
            depWords.append('not')
            
    return set(depWords),set(relWords)
    
def advmodFixer(relWords,depPairs):
    '''如果在清單內的形容詞有被副詞修飾，那就將那個形容詞改為 修飾該形容詞的副詞_該形容詞。
       此為網頁呈現之用。'''
    relWords=list(relWords)
    relWords_cp=relWords[:]
    for relWord in relWords_cp:
        for depPair in depPairs:
            if(relWord[0]==depPair[0]) and (depPair[2] == 'advmod'):
                #relWords.remove(relWord)
                relWords.append((depPair[1]+'_'+depPair[0],depPair[2]))
            if(relWord[0]==depPair[1]) and (depPair[2] == 'advmod'):
                #relWords.remove(relWord)
                relWords.append((depPair[0]+'_'+depPair[1],depPair[2]))
    return relWords

def sentAnalyzer(sent,target,message=0):
    '''句意分析主程式'''
    triples=stanfordDP(sent,0,message)
    depPairs=[(triple[0][0],triple[2][0],triple[1]) for triple in triples]    # [(word1, word2, relation),...]
                                                                              # i.e. ignore the type of words
                                                                              # preserve only the relations

    depWords,relWords=depRelWordsFind(target,depPairs)
    if(message!=0):
        print("dependent words=",depWords)
        print("related words=",relWords,'\n')
        
    #解決複合詞問題。以'noise level'為例，如果用'noise'去找解析樹內的相關詞，將找不到相關詞。需用以與其複合的詞，也就是'level'來查找，
    #方可找到相關詞。
    depWordsSet=[*depWords]
    relWordsSet=[*relWords]
    for relWord in relWords:
        if("compound" in relWord[1]):
            depWordsCmpPartner,relWordsCmpPartner=depRelWordsFind((relWord[0],'NN'),depPairs)
            if(message!=0):
                print("dependent words=",depWordsCmpPartner)
                print("related words=",relWordsCmpPartner,'\n')
            depWordsSet+=depWordsCmpPartner
            relWordsSet+=relWordsCmpPartner
            
    #depWordsSet=set(depWordsSet+[relWord[0] for relWord in relWordsSet])
    depWordsSet=set(depWordsSet)
    relWordsSet=set(relWordsSet)
    if(message!=0):
        print("***dependent words set=",depWordsSet)
        print("related words set=",relWordsSet)
    
    #將於relWordsSet內的形容詞加上副詞標籤，也就是將該形容詞改為 修飾該形容詞的副詞_該形容詞
    relWordsSet=advmodFixer(relWordsSet,depPairs)
    if(message!=0):
        print("related words set(advmod processed)=",relWordsSet)
    
    #去除複合關係等，因其不為形容詞，不應存於relWords清單內
    #relWordsFinal=[word for word in relWordsSet if (word[1]!='compound') \
    #    and (word[1]!='nmod') and (word[1]!='nsubj') and (word[1]!='dobj')]
    relWordsFinal=[word for word in relWordsSet if (word[1]!='compound')]
    #if there's something negative, it is strong and we will pick that word only
    ##relJJWordsFinal=[word for word in relWordsFinal if (word[1]=='JJ')]  
    #
    relNegAdvWordsFinal=[word for word in relWordsFinal if (word[1]=='neg' or word[1]=='advmod')]
    ##if (len(relJJWordsFinal)!=0) and (len(relNegWordsFinal)==0):
    ##    relWordsFinal=relJJWordsFinal
    if (len(relNegAdvWordsFinal)!=0):
        relWordsFinal=relNegAdvWordsFinal

    if(message!=0):
        print("***relWordsFinal=",relWordsFinal)
    
    return list(depWordsSet),relWordsFinal
    
def dicSentsCreat(dic_patterns,sentences):
    '''the goal of this method is to create 
    {label1:[label1_relatedSentence1,label1_relatedSentence2,...],
     label2:[label2_relatedSentence1,label2_relatedSentence2,...],...}'''

    dic_sents={}
    for key in dic_patterns:
        sents_list=[]
        for pattern in dic_patterns[key]:
            sents=findMatchedSentences(pattern,sentences)
            sents_list+=sents
        dic_sents[key]=list(set(sents_list))
        print("number of sentences of the label '%s' =%i"%(key,len(dic_sents[key])))
    return dic_sents

def dicParsedCreat(dic_patterns,dic_sents):
    '''the goal of this method is to create 
    {label1:[label1_parsedResult1,label1_parsedResult2,...],
     label2:[label2_parsedResult1,label2_parsedResult2,...],...}'''
    dic_parsed={}
    for key in dic_patterns:
        sents=dic_sents[key]
        wordsList=[]
        for idx,sent in enumerate(sents):
            print(idx)
            patterns=dic_patterns[key]
            for pattern in patterns:
                #print(key,pattern)
                    depWordsSet,relWordsSet=sentAnalyzer(sent[3],(pattern[0].strip(),pattern[1]),message=0)
                    wordsList.append((sent[3],list(depWordsSet),list(relWordsSet)))
                    break
        dic_parsed[key]=wordsList
    return dic_parsed

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def wCountClean(wCount,dic_patterns,label):

    #remove those keys which contain digits or '-'
    keysWithDigit=[key for key in wCount.keys() if hasNumbers(key) or ('-' in key)]
    for key in keysWithDigit:
        del wCount[key]
    #I'd like to take a look the non-processed data
    print(wCount)

    wCountCopy=dict(wCount)
    wCountFixed={} 
    # if the word is not_xxx, we add a minus sign to the freq of the word xxx
    for key in wCountCopy:
        if "not_" in key:
            wTail=key.split('_')[1]
            if(wTail!=[]):
                if nltk.pos_tag([wTail])[0][1]=='JJ':
                    negFreq=wCount[key]
                    wCountFixed[wTail]=-negFreq
                    del wCount[key]
                    for key2 in wCountCopy:
                        if key2==wTail:
                            wCountFixed[key2]=wCount[key2]-negFreq
        if "not_" not in key and "_" in key:
            wTail=key.split('_')[1]
            if(wTail!=[]):
                if(nltk.pos_tag([wTail])[0][1]=='JJ'):
                    Freq=wCount[key]
                    wCountFixed[wTail]=Freq
                    del wCount[key]
                    for key2 in wCountCopy:
                        if key2==wTail:
                            wCountFixed[key2]=wCount[key2]+Freq

    for key in wCount:
        # if the type of the word is JJ, we maintain it in the new dictionary wCountFixed
        if key not in wCountFixed: # those keys are processed already
            words=key.split("_")
            if(len(words)==1):
                wType=nltk.pos_tag(words)[0][1]
                if wType=='JJ':
                    wCountFixed[words[0]]=wCount[key]
                # fix an possible issue that the type of key is NN but it is actually JJ
                for item in dic_patterns[label]:
                    if words[0]==item[0].strip():
                        wType=item[1]
                        if(wType=='JJ'):
                            wCountFixed[words[0]]=wCount[key]
                            break         
    return wCountFixed

def wCountPlot(wCount):
    s = pd.Series(wCount, name='freqCount')
    s.plot.bar()
    plt.show()
    
def w2vecSVMFit():
    # load the trained model which is provided by Google
    model = Word2Vec.load_word2vec_format('/users/raemoen/SemanticProj/opMining/GoogleNews-vectors-negative300.bin', binary=True)
    # prepare the data for SVM
    df=pd.DataFrame.from_csv("data/training_vec.csv", encoding="utf-8",header=None)
    xtrain=np.array(df.iloc[:,1:])
    ytrain=np.array(df.iloc[:,0])
    # provide the data to SVM
    svm = SVC(kernel='linear', C=1.0, random_state=0)
    svm.fit(xtrain, ytrain)
    return model,svm

def emotDetermine(wCountFixed,model,svm):
    emotSum=0.
    if(wCountFixed=={}):
        emot=0
    else:
        for key in wCountFixed:
            try:
                vec=model[key]
                wEmot=svm.predict([vec])[0]
                wWeight=wCountFixed[key]*wEmot
                emotSum+=wWeight
                print('word=%15s, feq=%5i, emotion=%2i' %(key,wCountFixed[key],wEmot))
            except KeyError:
                print('Warning. Key %s is not in the Word2Vec model'%key)
                emot=0
        emot=1 if emotSum>0 else -1
    return(emot,emotSum)
    
    
    
    
model,svm=w2vecSVMFit()
#product types
prodTypes=["central","canister","handheld","robotic","stick","upright","wetdry"]

df=load_csv("2017-02-03")                               # load reviews from csv
summary(df)
reviews_list=store_prodRevs_in_a_list(prodTypes[-2],df)  # store reviews into a list
reviews_list=reviewsProcessing(reviews_list)            # the sentences of the reviews are cleaned

#product id
prodIDs='''B00KR5UJP4
'''
# obtain all the sentences of the reviews of the selected products
sentences=extractSentencesOfTheSelectedReviews(prodIDs,reviews_list)

#dic_patterns={'price':[(' price','NN'),(' expensive','JJ')],'switch':[(' switch','NN')]}

dic_patterns={'cord':[(' cord','NN'),(' cable','NN'),(' chord','NN'),(' wire','NN')],\
            'hose':[(' hose','NN')],\
            'nozzle': [(' nozzle','NN'), (' nossel','NN'), (' nozzel','NN')],\
            'assemble':[(' assemble','VB'),(' install','VB'),(' installation','NN')],\
            'price':[(' price','NN'),(' expensive','JJ'), (' cost ','NN')],\
            'suction':[(' suction','NN'), (' suck','VB')],\
            'battery':[(' battery','NN'),(' juice','NN'),(' lasting','JJ')],\
            'canister':[(' bag','NN'),(' canister','NN'),(' container','NN'),(' cylinder','NN'),(' recepticle','NN'),(' dispenser','NN'),(' filter','NN')],\
            'weight':[(' weight','NN'),(' heavy','JJ'),(' portable','JJ')],\
            'sound':[(' sound','NN'),(' loud','JJ'),(' quiet','JJ'),(' silent','JJ'),(' noise','NN'),(' noisy','JJ')],\
            'switch':[(' switch','NN'),(' on_off','NN')],\
            'smell':[(' smell','NN'), (' stink','VB'),(' fresh','JJ')],\
            'motor':[(' motor','NN'), (' powerful','JJ'),(' engine','NN')],\
            'vacuum':[(' vacuum','NN'),(' vaccum','NN'),(' vacuumin','NN'), (' vacumn','NN')] }


dic_sents=dicSentsCreat(dic_patterns,sentences)
dic_parsed=dicParsedCreat(dic_patterns,dic_sents)

wFreqList=[]
for key in dic_patterns:
    wCount=dict(collections.Counter([result[2][0][0] for result in dic_parsed[key] if len(result[2])==1 ]))
    wCountFixed=wCountClean(wCount,dic_patterns,key)
    print('key,wCountFixed=',key,wCountFixed)
    #wCountPlot(wCount)
    #wCountPlot(wCountFixed)
    emot=emotDetermine(wCountFixed,model,svm)
    print('emotion of the label %s=%i(%i)'%(key,emot[0],emot[1]))
    dependecy=list(wCountFixed.items())

    infoSingleLabel={"emotion": emot[0],"dependency":dependecy,"label":key}

    wFreqList.append(infoSingleLabel)
jsdata={"semantic":wFreqList,"pid":prodIDs[:-1]}
print(jsdata)

with open("result_%s.json"%prodIDs[:-1],'w+') as output:
    json.dump(jsdata,output)
