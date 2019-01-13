# bacterial_NER

Python3 Tensorflow==1.5.0

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
