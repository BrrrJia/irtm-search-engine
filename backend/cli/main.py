from ..core.indexing import InvertedIndex
from ..core import config
from ..core.retrieval import RetrievalEngine
from ..core.classification import NaiveBayesClassifier
from ..core.clustering import k_means, optimal_k_means, plot_clusters
from sklearn.preprocessing import normalize
import numpy as np
import argparse


def run_query_mode():
    print("Initializing Inverted Index System...\n")
    inv = InvertedIndex(config.DATA_PATH)
    inv.index()

    if config.USE_BIGRAM:
        print("Building bigram index...\n")
        inv.bigram_index()
    if config.USE_PERMUTERM:
        print("Building permuterm index...\n")
        inv.permuterm_index()

    print("Building TF-IDF matrix...\n")
    inv.build_tfidf_matrix()

    print("\nInitializing retrieval engine...\n")
    ret = RetrievalEngine(
        inv.term_dictionary,
        inv.postings_store,
        inv.df,
        inv.bigram_dictionary,
        inv.permuterm_dictionary,
        inv.term_to_index,
        inv.idf_vector,
        inv.tfidf_matrix,
    )

    while True:
        query_type = (
            input("\nEnter query type [term/tfidf/wildcard/exit]: ").strip().lower()
        )
        if query_type == "exit":
            break

        elif query_type not in ["term", "tfidf", "wildcard"]:
            print("Unsupported query type. Use one of: term / tfidf / wildcard / exit.")
            continue

        query = input("Enter query string: ").strip()

        if query_type == "term":
            terms = query.split()
            results = ret.terms_query(terms)
            ret.show_query_result(results)
        elif query_type == "tfidf":
            results = ret.query_similarity_top_k(query, k=config.TOP_K)
            ret.show_top_k_similar(results)

        elif query_type == "wildcard":
            mode = "bigram" if config.USE_BIGRAM else "permuterm"
            results = ret.wildcard_and_query(query, mode=mode)
            ret.show_query_result(results)


def run_classification():
    print("Initializing classification engine...\n")
    cls = NaiveBayesClassifier(config.TRAIN_PATH)

    print("Training classification engine...\n")
    cls.train()

    print("Testing classification engine...\n")
    cls.test(config.TEST_PATH)

    print("Evaluating classification engine...\n")
    cls.evaluate()


def run_clustering():
    print("Clustering data preparing...\n")
    inv_game = InvertedIndex(config.TRAIN_PATH)
    inv_game.index()
    inv_game.build_tfidf_matrix()
    data = normalize(inv_game.tfidf_matrix[:config.K_MEANS_DATA_SIZE], norm="l2")

    print("\nClustering beginns...\n")
    if config.USE_OPTIMAL_KMEANS:
        classes, centroids, _, rss = optimal_k_means(
            data, k=config.K_MEANS_K, n_init=config.K_MEANS_N_INIT
        )
        small_rss_cluster = np.where((rss > 0) & (rss < 1))[0]
        plot_clusters(
            data,
            classes,
            centroids=centroids,
            title="Optimal K-means Clustering",
            highlight=small_rss_cluster,
        )
    else:
        classes, centroids, _, rss = k_means(data, k=config.K_MEANS_K)
        small_rss_cluster = np.where((rss > 0) & (rss < 1))[0]
        plot_clusters(data, classes, centroids=centroids, highlight=small_rss_cluster)
    print("Clustering progress finished. Results saved to k-means.png")


def handle_noninteractive_mode(args):
    if args.task == "query":
        if not args.query_type or not args.query:
            print("Missing query_type or query string.")
            return

        # initialize index and retrieval engine
        # run tfidf / boolean term or wildcard query once and print result
        else:
            print("Query data initializing...\n")
            inv = InvertedIndex(config.DATA_PATH)
            inv.index()
            inv.build_tfidf_matrix()
            ret = RetrievalEngine(
                inv.term_dictionary,
                inv.postings_store,
                inv.df,
                inv.bigram_dictionary,
                inv.permuterm_dictionary,
                inv.term_to_index,
                inv.idf_vector,
                inv.tfidf_matrix,
            )
            print("\n")
            if args.query_type == "term":
                print("Boolean term query begins...\n")
                results = ret.terms_query(args.query.split())
                ret.show_query_result(results)
            elif args.query_type == "tfidf":
                print("TF-IDF similarity-based query begins...\n")
                results = ret.query_similarity_top_k(args.query, k=config.TOP_K)
                ret.show_top_k_similar(results)
            elif args.query_type == "wildcard":
                if config.USE_BIGRAM:
                    inv.bigram_index()
                    mode = "bigram"
                else:
                    inv.permuterm_index()
                    mode = "permuterm"
                print(f"Boolean {mode} wildcard query begins...\n")
                results = ret.wildcard_and_query(args.query, mode=mode)
                ret.show_query_result(results)
            return

    elif args.task == "classify":
        # run training + test + eval without interaction
        run_classification()

    elif args.task == "cluster":
        # perform clustering and save result
        run_clustering()


def main():
    parser = argparse.ArgumentParser(description="IRTM CLI Tool")
    parser.add_argument(
        "--task", choices=["query", "classify", "cluster"], help="Main task to perform"
    )
    parser.add_argument(
        "--query_type",
        choices=["term", "tfidf", "wildcard"],
        help="Query type for retrieval",
    )
    parser.add_argument("--query", type=str, help="Query string")
    args = parser.parse_args()

    # Branch 1: Argument-based (non-interactive mode)
    if args.task:
        handle_noninteractive_mode(args)
        return

    # Branch 2: Interactive CLI loop (default behavior)
    while True:
        task = input("Task [query/classify/cluster/exit]: ").strip().lower()
        if task == "exit":
            break
        elif task == "query":
            run_query_mode()
        elif task == "classify":
            run_classification()
        elif task == "cluster":
            run_clustering()
        else:
            print("Invalid task. Please try again.")


if __name__ == "__main__":
    main()
