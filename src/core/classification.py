from ..core.utils import preprocess, read_data
import math
from collections import defaultdict, Counter
from tqdm import tqdm

tqdm.pandas()


class NaiveBayesClassifier:
    def __init__(self, train_file):
        self.train_df = read_data(train_file, cls=True)
        self.test_df = None

        self.prior = {}
        self.tf_map = defaultdict(lambda: defaultdict(int))
        self.vocab = set()
        self.likelihood = defaultdict(Counter)

    def train(self):
        c_map = self.train_df["label"].value_counts(normalize=True)
        self.prior = c_map.to_dict()

        for label, group in self.train_df.groupby("label"):
            full_text = " ".join(
                group[col].fillna("").astype(str).str.cat(sep=" ")
                for col in ["name", "title", "review"]
            )
            terms = preprocess(full_text)
            self.tf_map[label] = Counter(terms)

        self.vocab = set(term for counter in self.tf_map.values() for term in counter)
        V_size = len(self.vocab)

        for label, counts in self.tf_map.items():
            total = sum(counts.values())
            default_p = 1 / (total + V_size)
            prob = Counter({t: (c + 1) / (total + V_size) for t, c in counts.items()})
            self.likelihood[label] = defaultdict(lambda: default_p, prob)

    def predict(self, tokens: list):
        best_cls = None
        best_log_prob = float("-inf")  # log space

        for cls in self.prior:
            log_prob = math.log(self.prior[cls])

            for t in tokens:
                p = self.likelihood[cls][t]
                log_prob += math.log(p)
            if log_prob > best_log_prob:
                best_log_prob = log_prob
                best_cls = cls
        return best_cls

    def test(self, test_file):
        self.test_df = read_data(test_file, cls=True)
        self.test_df["tokens"] = (
            self.test_df[["name", "title", "review"]]
            .fillna("")
            .agg(" ".join, axis=1)
            .apply(preprocess)
        )
        self.test_df["label_predicted"] = self.test_df["tokens"].apply(self.predict)

    def evaluate(self):
        confusion_matrix = {
            label: {"TP": 0, "FN": 0, "FP": 0}
            for label in self.test_df["label"].unique()
        }
        correct = 0

        for row in self.test_df.itertuples():
            true = row.label
            pred = row.label_predicted
            if true == pred:
                confusion_matrix[true]["TP"] += 1
                correct += 1
            else:
                confusion_matrix[true]["FN"] += 1
                confusion_matrix[pred]["FP"] += 1

        accuracy = correct / len(self.test_df)
        # print(f"Accuracy: {accuracy:.4f}")

        # Micro-averaged metrics
        micro_TP = sum(v["TP"] for v in confusion_matrix.values())
        micro_FP = sum(v["FP"] for v in confusion_matrix.values())
        micro_FN = sum(v["FN"] for v in confusion_matrix.values())

        micro_P = micro_TP / (micro_TP + micro_FP) if (micro_TP + micro_FP) else 0
        micro_R = micro_TP / (micro_TP + micro_FN) if (micro_TP + micro_FN) else 0
        micro_F1 = (
            2 * micro_P * micro_R / (micro_P + micro_R) if (micro_P + micro_R) else 0
        )

        # print(f"Micro F1: {micro_F1:.4f}")

        # Macro-averaged metrics
        precisions, recalls = [], []
        for v in confusion_matrix.values():
            p = v["TP"] / (v["TP"] + v["FP"]) if (v["TP"] + v["FP"]) else 0
            r = v["TP"] / (v["TP"] + v["FN"]) if (v["TP"] + v["FN"]) else 0
            precisions.append(p)
            recalls.append(r)

        macro_P = sum(precisions) / len(precisions)
        macro_R = sum(recalls) / len(recalls)
        macro_F1 = (
            2 * macro_P * macro_R / (macro_P + macro_R) if (macro_P + macro_R) else 0
        )

        print(f"Evaluation result:\nAccuracy: {accuracy:.4f}, F1: {macro_F1:.4f}")
        return accuracy, macro_F1
