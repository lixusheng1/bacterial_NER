# bacterial_NER

Python3 Tensorflow==1.5.0

#![image](https://github.com/lixusheng1/bacterial_NER/blob/master/1.png)

This project is used to recognize the bacteria in the text,the main structures of the model are CNN+BiLSTM+CRF and  domain fature(pos,dict)

The data format as the blew:

sentence	pos	dict	tag

Actinobacillus	NNP	B-bacteria	B-bacteria

actinomycetemcomitans	NNS	I-bacteria	I-bacteria

,	,	O	O

Porphyromonas	NNP	B-bacteria	B-bacteria

gingivalis	NN	I-bacteria	I-bacteria



The embedding file should in   data/embeddings/

The train/valid/test file should in data/dataset_name/

the pre_trained word embedding can download in  
glove: https://nlp.stanford.edu/projects/glove/

fastText:https://fasttext.cc/docs/en/english-vectors.html

PubMed2vec:http://bio.nlplab.org/#word-vector-tools

(1) Train 

python3 build_data.py

python3 train.py

(2) evaluate

python3 evaluate.py

(3) predict

python3 predict.py
