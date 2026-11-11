# ml-kit — API reference

**Version 1.0.0.** <div align="center">

Generated from the source in this repository: every name, argument and
note below is read out of the kit file, so it cannot describe a function
that is not there.

```js
<script src="ml-kit.js"></script>
MLKit.rng(...)
// or: const MLKit = require('./ml-kit.js')
```

## The surface (13 entries)

| Name | Arguments | What it does |
|---|---|---|
| `MLKit.rng` | `(seed)` | A seedable RNG (mulberry32): the same seed gives the same sequence, so a generated sentence or a sampled start is reproducible. |
| `MLKit.tfidf` | `(docs)` | — |
| `MLKit.search` | `(index, query, k)` | Query the index; returns [{i, score}] best first, dropping zero-score docs. |
| `MLKit.NaiveBayes` | `()` | — |
| `MLKit.knn` | `(points, k)` | — |
| `MLKit.kmeans` | `(points, k, opts)` | — |
| `MLKit.pca2` | `(points)` | — |
| `MLKit.linearRegression` | `(xs, ys)` | — |
| `MLKit.id3` | `(rows, attrs, target)` | — |
| `MLKit.predict` | `(tree, row)` | — |
| `MLKit.markov` | `(text, opts)` | — |
| `MLKit.generate` | `(model, opts)` | — |
| `MLKit.grammar` | `(rules)` | — |

## Notes

- Zero dependencies, one file, no build step: it works from `file://`, from a
  `<script>` tag and from Node.
- The version above is the `VERSION` file in this repository; a pinned copy is
  what an app vendors.
