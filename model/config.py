import os
from .general_utils import get_logger
from .data_utils import get_trimmed_glove_vectors, load_vocab,get_processing_word

class Config():
    def __init__(self, load=True):

        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            self.load()


    def load(self):

        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags  = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)
        self.vocab_poses = load_vocab(self.filename_poses)
        self.vocab_chunks = load_vocab(self.filename_chunks)


        self.nwords=len(self.vocab_words)
        self.nchars=len(self.vocab_chars)
        self.ntags=len(self.vocab_tags)
        self.nposes=len(self.vocab_poses)
        self.nchunks=len(self.vocab_chunks)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words,self.vocab_chars, lowercase=True)
        self.processing_tag  = get_processing_word(self.vocab_tags,lowercase=False, allow_unk=False)
        self.processing_pos = get_processing_word(self.vocab_poses, lowercase=False, allow_unk=False)
        self.processing_chunk = get_processing_word(self.vocab_chunks,lowercase=False, allow_unk=False)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_glove_vectors(self.filename_trimmed)if self.use_pretrained else None)
        self.pos_embeddings=get_trimmed_glove_vectors(self.filename_pos_trimmed)
        self.chunk_embedding=get_trimmed_glove_vectors(self.filename_chunk_trimmed)


    # general config
    dir_output = "results/"
    dir_model  = dir_output + "model.weights/"
    path_log   = dir_output + "log.txt"

    # embeddings
    dim_word = 300
    dim_char = 25
    dim_pos=25
    dim_chunk=5

    # glove files
    filename_glove = "data/embedding/word2vec.40B.{}d.txt".format(dim_word)
    # trimmed embeddings (created from glove_filename with build_data.py)
    filename_trimmed = "data/word2vec.40B.{}d.trimmed.npz".format(dim_word)
    filename_pos_trimmed="data/pos_one_hot.trimmed.npz"
    filename_chunk_trimmed="data/chunk_one_hot.trimmed.npz"
    use_pretrained = True

    # dataset
    filename_dev = "data/bacterial/dev_BIO.txt"
    filename_test = "data/bacterial/test_BIO.txt"
    filename_train = "data/bacterial/train_BIO.txt"


    max_iter = None # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    filename_words = "data/words.txt"
    filename_tags = "data/tags.txt"
    filename_chars = "data/chars.txt"
    filename_poses="data/poses.txt"
    filename_chunks="data/dict.txt"

    # training
    train_embeddings = False
    nepochs          = 100
    dropout          = 0.5
    batch_size       = 1
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    clip             = -1
    nepoch_no_imprv  = 5

    # model hyperparameters
    hidden_size_char = 25
    hidden_size_lstm = 100


    use_crf = True
    use_char_cnn=True 
    use_pos=True
    use_chunk=True


    filter_size=[3]
    filter_deep=30
