import argparse
import json

from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from tqdm import tqdm
from nltk.util import skipgrams, ngrams
import numpy as np

def main():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-path", type=str,
                        help="Path to the IMDB jsonl file.")
    parser.add_argument("--save-path", type=str,
                        help="Path to store the preprocessed corpus (output file name).")
    args = parser.parse_args()

    tokenizer = SpacyWordSplitter(language='en-core-web-sm')
    tokenized_examples = []
    with tqdm(open(args.data_path, "r")) as f:
        for line in f:
            example = {"text": line.strip()}
            tokens = list(map(str, tokenizer.split_words(example['text'])))
            example['text'] = ' '.join(tokens)
            tokenized_examples.append(example)
    write_jsons_to_file(tokenized_examples, args.save_path)


def write_jsons_to_file(jsons, save_path):
    """
    Write each json object in 'jsons' as its own line in the file designated by 'save_path'.
    """
    # Open in appendation mode given that this function may be called multiple
    # times on the same file (positive and negative sentiment are in separate
    # directories).
    out_file = open(save_path, "a")
    for example in tqdm(jsons):
        json.dump(example, out_file, ensure_ascii=False)
        out_file.write('\n')

if __name__ == '__main__':
    main()
