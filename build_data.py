from model.config import Config
from model.data_utils import CoNLLDataset, get_vocabs, UNK, NUM, \
    get_glove_vocab, write_vocab, load_vocab, get_char_vocab, \
    export_trimmed_glove_vectors, get_processing_word,export_trimed_ont_hot_vectors


def main():
    # get config and processing of words
    config = Config(load=False)
    processing_word = get_processing_word(lowercase=True)
    processing_pos=get_processing_word()
    processing_chunk=get_processing_word()
    # Generators
    dev   = CoNLLDataset(config.filename_dev, processing_word,processing_pos,processing_chunk)
    test  = CoNLLDataset(config.filename_test, processing_word,processing_pos,processing_chunk)
    train = CoNLLDataset(config.filename_train, processing_word,processing_pos,processing_chunk)

    # Build Word and Tag vocab
    vocab_words, vocab_tags,vocab_poses,vocab_chunks = get_vocabs([train, dev, test])
    vocab_glove = get_glove_vocab(config.filename_glove)

    vocab = [i  for i in vocab_words if i in vocab_glove]
    vocab.append(UNK)
    vocab.append(NUM)
    vocab.append("$pad$")
    vocab_poses.append("$pad$")
    vocab_chunks.append("$pad$")
    vocab_tags.append("$pad$")

    # Save vocab
    write_vocab(vocab, config.filename_words)
    write_vocab(vocab_tags, config.filename_tags)
    write_vocab(vocab_poses, config.filename_poses)
    write_vocab(vocab_chunks, config.filename_chunks)

    # Trim GloVe Vectors	
    vocab = load_vocab(config.filename_words)
    print(len(vocab))
    export_trimmed_glove_vectors(vocab, config.filename_glove,
                                config.filename_trimmed, config.dim_word)

    vocab=load_vocab(config.filename_poses)
    export_trimed_ont_hot_vectors(vocab,config.filename_pos_trimmed)

    vocab=load_vocab(config.filename_chunks)
    export_trimed_ont_hot_vectors(vocab,config.filename_chunk_trimmed)

    # Build and save char vocab
    train = CoNLLDataset(config.filename_train)
    vocab_chars = get_char_vocab(train)
    vocab_chars.append("$pad$")
    write_vocab(vocab_chars, config.filename_chars)


if __name__ == "__main__":
    main()
