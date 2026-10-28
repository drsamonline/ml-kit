<div align="center">

# ml-kit

[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-2e7d32)](LICENSE)

**The classic algorithms, from scratch — for offline features.**

tf-idf · Naive Bayes · k-NN · k-means · PCA · ID3 · linear regression · Markov chains · grammar

A single file, zero dependencies, no model, no API key, no network. MIT.

</div>

---

## THE BOUNDARY — read this first

This kit is for **ranking, similarity, clustering and simple trend**. It is **NOT a
diagnostic or predictive engine**, and **nothing in it may be used to claim a
clinical or medical capability.** Several apps that use this kit are clinical; the
algorithms here are generic maths and make no clinical claim. A model that ranks
notes, groups similar cases or draws a trend line is not a diagnosis, and must
never be presented as one.

## Why from scratch

Forge's rule is offline, single-file, zero-dependency — so the usual ML frameworks
(TensorFlow, Spark, scikit-learn) are unusable: they are hundreds of megabytes and
need a runtime. But the algorithms that actually earn their place offline are the
*classic* ones, and each fits in tens of lines. This is the light half of Forge's AI
story: **heavy AI uses the user's own key; light AI runs on the device, from scratch.**

## What's in it

| Algorithm | For |
|---|---|
| `tfidf` + `search` | ranked text search (tf-idf weighting, cosine similarity) |
| `NaiveBayes` | a tiny bag-of-words classifier (tag a note, categorise an entry) |
| `knn` | "items like this one" |
| `kmeans` | clustering |
| `pca2` | reduce many dimensions to two, for plotting |
| `id3` + `predict` | a small, readable rules classifier |
| `linearRegression` | a trend line |
| `markov` + `generate` | statistical generation from a corpus (char- or word-level) |
| `grammar` | rule-based generation: rules, slots, weighted alternatives |

Plus `rng(seed)` — a seedable RNG, so a sampled start or a generated sentence is
reproducible.

## Usage

```js
var ML = require("./ml-kit.js");   // Node; in a browser it is window.MLKit

// Ranked search
var index = ML.tfidf(["the cat sat on the mat", "dogs bark and cats meow"]);
ML.search(index, "cat");           // -> [{ i: 0, score: ... }]

// Classify
var nb = ML.NaiveBayes().train([
  { text: "good product", label: "pos" },
  { text: "terrible waste", label: "neg" }
]);
nb.classify("excellent product");  // -> "pos"

// A trend line
ML.linearRegression([0, 1, 2, 3], [1, 3, 5, 7]).slope;   // -> 2

// Reproducible generation
var m = ML.markov(corpus, { order: 2, word: true });
ML.generate(m, { seed: 42, length: 12 });                // same seed -> same text
```

## Tests

```
node test-node.js
```

A known tf-idf ranking case, a Naive Bayes classification case, a k-means
convergence case, a regression line against a known slope, PCA, k-NN, ID3, and
seeded-reproducibility checks for the Markov chain and the grammar. CI runs it on
every push.

## Design notes

- **Every algorithm is reimplemented from the idea**, not copied. Where a survey
  suggested a reference, only the shape was taken — the code is ours.
- **Deterministic where it matters.** k-means seeds from the first *k* points;
  Markov and the grammar take a seed, so the same seed gives the same output.
- **Small and readable.** Tens of lines each, so the maths can be read and checked.
- **UMD.** `module.exports` in Node, `MLKit` in a browser — the same shape as the
  other kits.

## License

PolyForm Noncommercial — see [LICENSE](LICENSE). © 2026 Dr Sohil Momin.
