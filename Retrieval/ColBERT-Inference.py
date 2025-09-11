import argparse
import os
import json
import random
import csv
import time
from tqdm.auto import tqdm
from datasets import load_dataset

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate ColBERT retrieval on CRIC dataset")
    parser.add_argument('--subset', type=int, default=100, help="Subset of CRIC questions to use")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to ColBERT checkpoint")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to Dataset split file")
    parser.add_argument('--output', type=str, default='./bufferFile2.txt', help="Output file for results")
    return parser.parse_args()

def main(args):
    os.environ["NCCL_TIMEOUT_MS"] = "1200000"
    os.environ['MKL_THREADING_LAYER'] = 'GNU'

    print('\nLoading Commonsense Corpus from Huggingface')
    omcs_with_embeddings = load_dataset("dutta18/omcs_commonsense_corpus1.5M_for_fast_NN_search", split='train')
    omcs1_5MList = list(set(omcs_with_embeddings['text'][:500]))
    random.shuffle(omcs1_5MList)

    # Write collection to TSV
    with open('omcs1.5M.tsv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        for index, string in enumerate(omcs1_5MList):
            writer.writerow([index, string])

    # Load CRIC test set
    # ../cric/test_v1_questions.json'
    with open(args.dataset_path, "r") as file:
        test_json = json.load(file)

    questionList_test, k_triplet_test = [], []
    for pointer in test_json:
        questionList_test.append(pointer['question'])
        k_triplet_test.append(' '.join(pointer['sub_graph']['knowledge_items'][0]['triplet']) + '.')

    # Subset
    questionList_test = questionList_test[:args.subset]
    k_triplet_test = k_triplet_test[:args.subset]

    # Write queries to TSV
    with open('cricQuestions.tsv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        for index, string in enumerate(questionList_test):
            writer.writerow([index, string])

    # Run indexing
    with Run().context(RunConfig(nranks=1, experiment='evaluation_exps')):
        config = ColBERTConfig(doc_maxlen=300, nbits=2)
        indexer = Indexer(checkpoint=args.checkpoint, config=config)
        indexer.index(name='colbert_OMCS.2bits', collection=Collection(path='omcs1.5M.tsv'), overwrite=True)

    # Run search
    with Run().context(RunConfig(experiment='evaluation_exps')):
        searcher = Searcher(index='colbert_OMCS.2bits')
        k = 5
        with open(args.output, 'w') as file:
            for query in tqdm(questionList_test):
                file.write(f"#> {query}\n")
                results = searcher.search(query, k=k)
                for pid, rank, score in zip(*results):
                    file.write(f"\t [{rank}] \t\t {score:.1f} \t\t {searcher.collection[pid]}\n")

    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
