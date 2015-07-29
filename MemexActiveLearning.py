from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
from BeautifulSoup import BeautifulSoup
import json

class MemexActiveLearner:

    def __init__(self):
        self.text=None
        self.clean_text = None
        self.clf = LogisticRegression(C=100, penalty='l1')
        self.next_idx=None
        self.next_label=None
        

    def load_pennystocktweet_news(self):
        clean_text = [line.rstrip() for line in open('data/spam_train.txt','r')]
        #self.text = self.clean_text
        
        result = [json.loads(line) for line in open('data/promotions/3.jl')]
        #load uncleaned data
        self.text = [obj['content'] for obj in result if obj['_type']=='news']
        del result
        
        ##1 is positive, 0 is unlabeled
        self.my_labels = np.array([int(line.rstrip()) for line in open('data/spam_labels.csv','r')])
        #set 12 to -1 
        self.my_labels[12]=-1
        #zero_indices = np.where(my_labels==0)[0]
        self.featurize_text(clean_text)
        
    def featurize_text(self,clean_text):
        vocab=[line.split(',')[0] for line in open('data/spam_stability_selection.csv','r')][:1000]
        vectorizer = CountVectorizer( stop_words="english", ngram_range=(1, 1),vocabulary=vocab, analyzer="word", max_df=0.8, min_df=0.01, binary=True,max_features=None)
        self.X = vectorizer.fit_transform(clean_text)
        
    def find_closest(self,array,value):
        idx = (np.abs(array-value)).argmin()
        return idx

    def find_highest(self,array):
        idx = array.argmax()
        return idx

    def run_classifier(self,init=False):
        if not init:
            if self.next_idx is None:
                print 'warning: no observation has been selected to receive a label'
                return
            if self.next_label is None:
                print 'warning: an observation has been selected to receive a label, but the label has not been set'
                return 
        self.next_idx=None
        self.next_label=None
        indices = np.concatenate((np.where(self.my_labels==-1)[0],np.where(self.my_labels==1)[0]))
        self.test_indices = np.where(self.my_labels==0)[0]
        tmp_X=self.X[indices]
        tmp_labels = self.my_labels[indices]
        tmp_Xtest=self.X[self.test_indices]
        self.clf.fit(tmp_X,tmp_labels)
        self.predictions=self.clf.predict_proba(tmp_Xtest)

        
    def get_next(self,uncertainty=True):
        if self.predictions is not None:
            if self.next_idx is not None:
                print 'Next observation already selected'
            else:
                if uncertainty:
                    next_idx=self.find_closest(self.predictions[:,1],0.5)
                else:
                    #get highest for class that is underrepresented:
                    mysum=self.my_labels.sum()
                    if mysum>=0:
                        next_idx=self.find_highest(self.predictions[:,0])
                    else:
                        next_idx=self.find_highest(self.predictions[:,1])
        
                self.next_idx=self.test_indices[next_idx]
                                    
            print self.text[self.next_idx]
        else:
            print 'classifier has not yet been trained'
        
    def clean_text(self,corpus):
        #unwanted chars:
        #to be replaced with ''
        charsNOWHITE = '"#()\'*+/<=>@[\\]^_`{|}~'
        #to be replaced with ' '
        charsWHITE =',.&!+:;?\n'
        remove_nowhite_map = dict((ord(char), None) for char in charsNOWHITE)
        remove_white_map = dict((ord(char), u' ') for char in charsWHITE)
        #tab1=' '*len(charsWHITE)
        #transtable=string.maketrans(charsWHITE,tab1)
        for line in corpus:
            #remove non ascii characters
            line = filter(lambda x: x in string.printable, line)
            parsed_html = BeautifulSoup(line)
            texts = parsed_html.findAll(text=True)
            line=" ".join(texts).lower()

            #or replace non ascii characters
            #html = ''.join([x.lower() if (x in string.printable) else 'X' for x in line])
            #line = re.sub(r'X[X\s]*',' X ',line)

            #remove certain characters using fast c code translate function
            line=line.translate(remove_nowhite_map)
            #replace certain characters with whitesapce
            line=line.translate(remove_white_map)
            
            #If text is string and not unicode, use this:
            #line=line.translate(None,charsNOWHITE)
            #line=line.translate(transtable)

            #remove excessive whitespaces
            #line = " ".join(line.split())

            yield line.lower()

        
    def set_label(self,value):
        if value not in [-1,1,'-1','1']:
            print 'invalid label'
            return
        if self.next_label is not None:
            print 'label has already been set'
        else:
            self.next_label=value
            self.my_labels[self.next_idx]=value

    def clean_text(self,corpus,html=True):
        #unwanted chars:
        #to be replaced with ''
        charsNOWHITE = '"#()\'*+/<=>@[\\]^_`{|}~'
        #to be replaced with ' '
        charsWHITE =',.&!+:;?\n'
        remove_nowhite_map = dict((ord(char), None) for char in charsNOWHITE)
        remove_white_map = dict((ord(char), u' ') for char in charsWHITE)
        #tab1=' '*len(charsWHITE)
        #transtable=string.maketrans(charsWHITE,tab1)
        if html:
            for line in corpus:
                #remove non ascii characters
                line = filter(lambda x: x in string.printable, line)
                parsed_html = BeautifulSoup(line)
                texts = parsed_html.findAll(text=True)
                line=" ".join(texts).lower()

                #or replace non ascii characters
                #html = ''.join([x.lower() if (x in string.printable) else 'X' for x in line])
                #line = re.sub(r'X[X\s]*',' X ',line)

                #remove certain characters using fast c code translate function
                line=line.translate(remove_nowhite_map)
                #replace certain characters with whitesapce
                line=line.translate(remove_white_map)

                #If text is string and not unicode, use this:
                #line=line.translate(None,charsNOWHITE)
                #line=line.translate(transtable)

                #remove excessive whitespaces
                #line = " ".join(line.split())

                yield line
        else:
            for line in corpus:
                #remove non ascii characters
                line = filter(lambda x: x in string.printable, line)

                #or replace non ascii characters
                #html = ''.join([x.lower() if (x in string.printable) else 'X' for x in line])
                #line = re.sub(r'X[X\s]*',' X ',line)

                #remove certain characters using fast c code translate function
                line=line.translate(remove_nowhite_map)
                #replace certain characters with whitesapce
                line=line.translate(remove_white_map)

                #If text is string and not unicode, use this:
                #line=line.translate(None,charsNOWHITE)
                #line=line.translate(transtable)

                #remove excessive whitespaces
                #line = " ".join(line.split())

                yield line.lower()
