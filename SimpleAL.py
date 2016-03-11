

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import numbers
import numpy as np
from BeautifulSoup import BeautifulSoup
import json
import os
import cPickle as pickle
from itertools import izip
import string

class SimpleAL:

    def __init__(self):
        self.text=None
        self.clf = LogisticRegression(C=100, penalty='l1')
        self.next_idx=None
        self.vectorizer=None
        self.binary=True
        self.max_df=0.8
        self.min_df=0.01
        self.max_features=30000
        self.keys=None
        self.labels_unique=None

    def init_data(self,corpus,labels,clean=False):
        #corpus is a list of text
        #labels is a numpy array where 0 is unlabeled, 1 is positive, -1 is negative
        self.text = corpus
        self.my_labels=labels
        self.labels_unique=np.unique(labels)
        #remvove 0 since it denotes unlabeled
        self.labels_unique=self.labels_unique[self.labels_unique!=0]
        if clean:
            self.featurize_text(corpus)
        else:
            
            clean_text = [line for line in self.clean_text(corpus)]
            self.featurize_text(clean_text)

    def featurize_text(self,clean_text,vectorizer_preload=False):
        if vectorizer_preload:
            #vectorizer already loaded, don't fit
            self.X = self.vectorizer.transform(clean_text)
            return 
        else:
            self.vectorizer = CountVectorizer( stop_words="english", ngram_range=(1, 1), analyzer="word", binary=True,max_features=self.max_features)
            vocabulary, X = self.vectorizer._count_vocab(clean_text, None)
            if self.binary:
                X.data.fill(1)
            self.X_full = self.vectorizer._sort_features(X, vocabulary)

        self.vocabulary_ = vocabulary
        self.limit_features()

    def limit_features(self):
        max_df = self.max_df
        min_df = self.min_df
        max_features = self.max_features

        n_doc = self.X_full.shape[0]
        max_doc_count = (max_df
                         if isinstance(max_df, numbers.Integral)
                         else max_df * n_doc)
        min_doc_count = (min_df
                         if isinstance(min_df, numbers.Integral)
                         else min_df * n_doc)
        if max_doc_count < min_doc_count:
                raise ValueError("max_df corresponds to < documents than min_df")
        self.X,stop_words_ = self.vectorizer._limit_features(self.X_full, self.vocabulary_,max_doc_count,min_doc_count,max_features)

    def change_cutoffs(self,max_df=None,min_df=None):
        flag=False
        if max_df is not None:
            if float(max_df) != float(self.max_df):
                self.max_df=max_df
                flag=True
        if min_df is not None:
            if float(min_df) != float(self.min_df):
                self.min_df = min_df
                flag=True
        if flag==True:
            self.limit_features()     
 
    def margin_sampling(self,array):
        # Margin sampling: 
        # Scheffer, T., Decomain, C., & Wrobel, S. (2001). Active hidden markov models for information extraction. 
        #    In Advances in Intelligent Data Analysis (pp. 309-318). Springer Berlin Heidelberg.
        desc=(-np.sort(-array,axis=1))#sort rows of predictions
        return np.argmin(desc[:,1]-desc[:,2])

    def find_highest(self,array):
        idx = array.argmax()
        return idx

    def run_classifier(self,init=False):
        self.next_idx=None
        indices = np.where(self.my_labels!=0)[0]
        self.test_indices = np.where(self.my_labels==0)[0]
        tmp_X=self.X[indices]
        tmp_labels = self.my_labels[indices]
        tmp_Xtest=self.X[self.test_indices]
        self.clf.fit(tmp_X,tmp_labels)
        self.predictions=self.clf.predict_proba(tmp_Xtest)

        
    def set_next(self,uncertainty=True):
        if self.predictions is not None:
            if self.next_idx is None:
                if uncertainty:
                    next_idx=self.margin_sampling(self.predictions)
                else:
                    #get highest for class that is underrepresented:
                    counts=np.bincount(labels.astype(np.int))
                    mysum=self.my_labels.sum()
                    lblindx=np.argmax(counts[1:])
                    next_idx=self.find_highest(self.predictions[:,lblindx])
                    next_idx=self.find_highest(self.predictions[:,lblindx])
                #now that we have selected this for labeling, remove it
                self.predictions = np.delete(self.predictions, (next_idx), axis=0)
                self.next_idx=self.test_indices[next_idx]
                self.test_indices=np.delete(self.test_indices, (next_idx), axis=0)
                return True
            else:
                return False
        else:
            return False
                                    
        
    def load_classifier(self,filename,path=''):
        if not os.path.isfile(f):
            print 'File does not exists'
            return
        self.clf=pickle.load(open(f,'wb'))
        
    def load_vectorizer(self,filename,path=''):
        if not os.path.isfile(f):
            print 'File does not exists'
            return
        self.vectorizer=pickle.load(open(f,'wb'))

    def save_classifier(self,filename,path='',overwrite=False):
        f=os.path.join(path,filename)
        if not overwrite:
            if os.path.isfile(f):
                print 'File already exists'
                return
        pickle.dump(self.clf,open(f,'wb'))


    def save_vectorizer(self,filename,path='',overwrite=False):
        f=os.path.join(path,filename)
        if not overwrite:
            if os.path.isfile(f):
                print 'File already exists'
                return
        pickle.dump(self.vectorizer,open(f,'wb'))
            
        
    def set_label(self,value):
        if value not in self.labels_unique:
            print 'invalid label'
            return
        if self.next_idx is not None:
            self.my_labels[self.next_idx]=value
            self.next_idx=None
        else:
            print 'no index to be labeled selected. Call set_next()'

    def load_marketwired(self,dirs=['marketwired_data_rerun2','marketwired_data_rerun1','marketwired_data_symbolwise']):
        for path in dirs:
            if os.path.exists(path):
                for fname in os.listdir(path):
                    pathname=os.path.join(path,fname)
                    f=open(pathname,'r')
                    yield pathname,f.read()                
                    f.close()
            else:
                print 'directory '+path+' does not exist. Data not loaded'

    def clean_text(self,corpus,html=False):
        #unwanted chars:
        #to be replaced with ''
        charsNOWHITE = '"#()\'*+/<=>@[\\]^_`{|}~'
        #to be replaced with ' '
        charsWHITE =',.&!+:;?\n'
        remove_nowhite_map = dict((ord(char), None) for char in charsNOWHITE)
        remove_white_map = dict((ord(char), u' ') for char in charsWHITE)
        #
        strp=frozenset(string.printable)
        tab1=' '*len(charsWHITE)
        transtable=string.maketrans(charsWHITE,tab1)
        if html:
            for line in corpus:
                #remove non ascii characters
                line = filter(lambda x: x in strp, line)
                parsed_html = BeauifulSoup(line)
                texts = parsed_html.findAll(text=True)
                line=" ".join(texts).lower()

                #or replace non ascii characters
                #html = ''.join([x.lower() if (x in string.printable) else 'X' for x in line])
                #line = re.sub(r'X[X\s]*',' X ',line)

                #remove certain characters using fast c code translate function
                #line=line.translate(remove_nowhite_map)
                #replace certain characters with whitesapce
                #line=line.translate(remove_white_map)

                #If text is string and not unicode, use this:
                line=line.translate(None,charsNOWHITE)
                line=line.translate(transtable)

                #remove excessive whitespaces
                #line = " ".join(line.split())

                yield line
        else:
            for line in corpus:
                #remove non ascii characters
                line = filter(lambda x: x in strp, line)

                #remove certain characters using fast c code translate function
                #line=line.translate(remove_nowhite_map)
                #line=line.translate(remove_white_map)
                line=line.translate(None,charsNOWHITE)
                line=line.translate(transtable)


                yield line.lower()

from flask import Flask,render_template,request
app = Flask(__name__)
app.debug=True
@app.route('/',methods=['GET', 'POST'])
def classify():
    if 'classlabel' in request.args: 
        SAL.set_label(int(request.args.get('classlabel')))

    if 'retrain' in request.args: 
        SAL.run_classifier()
        

    if SAL.next_idx is None:
        if SAL.set_next():
            text = SAL.text[SAL.next_idx]
        else:
            text = 'Next index could not be selecteNext index could not be selected'
    else:
        text = SAL.text[SAL.next_idx]

    return render_template('index.html',buttons=class_buttons,
                           mytext=text.decode('utf-8'))

if __name__ == "__main__":  
    SAL=SimpleAL()
    #corpus=[(key,doc) for key,doc in SAL.load_marketwired()]
    corpus=[(key,doc) for key,doc in SAL.load_marketwired(dirs=['TESTDATA'])]
    keys,corpus=izip(*corpus)
    labels=np.zeros(len(keys))
    pos_indices=np.array([i for i,text in enumerate(corpus) if 'forward-looking' in text])
    labels[pos_indices]=1
    neg_indices=np.array([3])
    nothing_ind=np.array([1,2,4])
    labels[neg_indices]=2
    labels[nothing_ind]=3
    SAL.init_data(corpus,labels)   
    SAL.run_classifier(init=True)
    #import IPython
    #IPython.embed() 
    class_buttons=[('Spam',1,''),('NOT spam',2,''),('Ambiguous',3,'')]
    app.run()
