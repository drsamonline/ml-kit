/* Tests for ml-kit. Run: node test-node.js */
"use strict";
var ML = require("./ml-kit.js");
var pass = 0, fail = 0;
function ok(name, cond) { if (cond) { pass++; console.log("PASS  " + name); } else { fail++; console.log("FAIL  " + name); } }
function near(a, b, t) { return Math.abs(a - b) <= (t || 1e-6); }

/* ---- tf-idf ranking ---- */
var docs = [
  "the cat sat on the mat",
  "dogs bark and cats meow",
  "quantum physics and general relativity"
];
var idx = ML.tfidf(docs);
var res = ML.search(idx, "cat");
ok("tf-idf: 'cat' ranks the cat document first", res.length > 0 && res[0].i === 0);
ok("tf-idf: the irrelevant document scores zero", res.every(function (r) { return r.i !== 2; }));
var res2 = ML.search(idx, "dogs meow");
ok("tf-idf: 'dogs meow' ranks the dogs document first", res2[0].i === 1);

/* ---- Naive Bayes ---- */
var nb = ML.NaiveBayes().train([
  { text: "good product", label: "pos" },
  { text: "excellent quality product", label: "pos" },
  { text: "terrible waste", label: "neg" },
  { text: "awful waste of money", label: "neg" }
]);
ok("naive bayes: 'excellent product' -> pos", nb.classify("excellent product") === "pos");
ok("naive bayes: 'terrible waste' -> neg", nb.classify("terrible waste") === "neg");

/* ---- k-NN ---- */
var knn = ML.knn([{ x: [0, 0], label: "A" }, { x: [1, 1], label: "A" }, { x: [10, 10], label: "B" }, { x: [11, 11], label: "B" }], 3);
ok("k-NN: [0.5,0.5] -> A", knn([0.5, 0.5]) === "A");
ok("k-NN: [10.5,10.5] -> B", knn([10.5, 10.5]) === "B");

/* ---- k-means convergence ---- */
var km = ML.kmeans([[0, 0], [0, 1], [1, 0], [10, 10], [10, 11], [11, 10]], 2);
ok("k-means: converges within 100 iterations", km.iterations < 100);
ok("k-means: the two groups are separated", JSON.stringify(km.labels) === JSON.stringify([0, 0, 0, 1, 1, 1]));

/* ---- linear regression ---- */
var lr = ML.linearRegression([0, 1, 2, 3], [1, 3, 5, 7]);
ok("regression: slope is 2", near(lr.slope, 2));
ok("regression: intercept is 1", near(lr.intercept, 1));
ok("regression: predict(4) = 9", near(lr.predict(4), 9));

/* ---- PCA ---- */
var pca = ML.pca2([[0, 0], [1, 2], [2, 4], [3, 6]]);
var c0 = pca.components[0];
ok("PCA: first component points along y=2x", near(Math.abs(c0[1] / c0[0]), 2, 1e-3));
ok("PCA: projects to 2 dimensions", pca.points[0].length === 2);

/* ---- ID3 ---- */
var tree = ML.id3([
  { a: "x", b: "1", c: "yes" },
  { a: "x", b: "2", c: "yes" },
  { a: "y", b: "1", c: "no" },
  { a: "y", b: "2", c: "no" }
], ["a", "b"], "c");
ok("ID3: splits on the informative attribute", tree.attr === "a");
ok("ID3: predicts x -> yes", ML.predict(tree, { a: "x", b: "2" }) === "yes");
ok("ID3: predicts y -> no", ML.predict(tree, { a: "y", b: "1" }) === "no");

/* ---- Markov ---- */
var corpus = "the quick brown fox jumps over the lazy dog the quick brown fox sleeps";
var mm = ML.markov(corpus, { order: 2, word: true });
var g1 = ML.generate(mm, { seed: 42, length: 12 });
var g2 = ML.generate(mm, { seed: 42, length: 12 });
ok("markov: same seed -> same text", g1 === g2);
ok("markov: text is non-empty", g1.length > 0);
var vocab = {}; corpus.split(/\s+/).forEach(function (w) { vocab[w] = 1; });
ok("markov: every generated word is from the corpus", g1.split(/\s+/).every(function (w) { return vocab[w]; }));

/* ---- Grammar ---- */
var gr = ML.grammar({
  greeting: ["Hello, #name#!", { text: "Hi #name#", weight: 0.5 }],
  name: ["world", "there"]
});
var s1 = gr.expand("greeting", 7), s2 = gr.expand("greeting", 7);
ok("grammar: same seed -> same sentence", s1 === s2);
ok("grammar: slots are expanded", /world|there/.test(s1) && !/#/.test(s1));

console.log("\n" + pass + " passed, " + fail + " failed");
process.exit(fail ? 1 : 0);
