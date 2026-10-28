/* ml-kit — the classic algorithms, from scratch, for offline features. Zero dependencies.
 *
 * tf-idf + cosine, Naive Bayes, k-NN, k-means, PCA, ID3, linear regression, Markov
 * chains and a grammar expander — each reimplemented in tens of lines. No model, no
 * API key, no network, no WASM. UMD: `module.exports` in Node, `MLKit` in a browser.
 *
 * THE BOUNDARY: this kit is for RANKING, SIMILARITY, CLUSTERING and simple TREND.
 * It is NOT a diagnostic or predictive engine, and nothing in it may be used to
 * claim a clinical or medical capability. Several apps that use it are clinical;
 * the algorithms here are generic and make no clinical claim.
 *
 * PolyForm Noncommercial 1.0.0. (c) 2026 Dr Sohil Momin.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.MLKit = factory();
}(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var ML = { version: "0.1.0" };

  /* ------------------------------------------------------------------ utils */

  // A seedable RNG (mulberry32): the same seed gives the same sequence, so a
  // generated sentence or a sampled start is reproducible.
  ML.rng = function (seed) {
    var a = (seed >>> 0) || 0x9e3779b9;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };

  function words(text) { return String(text).toLowerCase().match(/[a-z0-9']+/g) || []; }
  function euclid(a, b) {
    var s = 0;
    for (var i = 0; i < a.length; i++) { var d = a[i] - b[i]; s += d * d; }
    return Math.sqrt(s);
  }

  /* ------------------------------------------------------- tf-idf + cosine */
  // Ranked text search. search-kit uses this as its ranking.

  ML.tfidf = function (docs) {
    var n = docs.length;
    var tfs = docs.map(function (d) {
      var toks = words(d), m = {};
      toks.forEach(function (t) { m[t] = (m[t] || 0) + 1; });
      return { m: m, len: toks.length || 1 };
    });
    var df = {};
    tfs.forEach(function (tf) { Object.keys(tf.m).forEach(function (t) { df[t] = (df[t] || 0) + 1; }); });
    var idf = {};
    Object.keys(df).forEach(function (t) { idf[t] = Math.log((1 + n) / (1 + df[t])) + 1; });
    var vecs = tfs.map(function (tf) {
      var v = {};
      Object.keys(tf.m).forEach(function (t) { v[t] = (tf.m[t] / tf.len) * idf[t]; });
      return v;
    });
    return { idf: idf, docs: vecs, n: n };
  };

  function cosine(a, b) {
    var dot = 0, na = 0, nb = 0, k;
    for (k in a) { na += a[k] * a[k]; if (b[k] !== undefined) dot += a[k] * b[k]; }
    for (k in b) { nb += b[k] * b[k]; }
    return (na && nb) ? dot / Math.sqrt(na * nb) : 0;
  }

  // Query the index; returns [{i, score}] best first, dropping zero-score docs.
  ML.search = function (index, query, k) {
    var qt = words(query), qm = {}, len = qt.length || 1;
    qt.forEach(function (t) { qm[t] = (qm[t] || 0) + 1; });
    var qv = {};
    Object.keys(qm).forEach(function (t) { if (index.idf[t] !== undefined) qv[t] = (qm[t] / len) * index.idf[t]; });
    var scored = index.docs.map(function (v, i) { return { i: i, score: cosine(qv, v) }; });
    scored.sort(function (a, b) { return b.score - a.score; });
    return scored.slice(0, k || index.docs.length).filter(function (s) { return s.score > 0; });
  };

  /* ----------------------------------------------------------- Naive Bayes */
  // A tiny bag-of-words classifier: tag a note, categorise an entry.

  ML.NaiveBayes = function () {
    var counts = {}, totals = {}, classCounts = {}, vocab = {}, docs = 0;
    function train(items) {
      items.forEach(function (it) {
        var label = it.label;
        classCounts[label] = (classCounts[label] || 0) + 1; docs++;
        counts[label] = counts[label] || {}; totals[label] = totals[label] || 0;
        words(it.text).forEach(function (t) { counts[label][t] = (counts[label][t] || 0) + 1; totals[label]++; vocab[t] = 1; });
      });
      return api;
    }
    function classify(text) {
      var toks = words(text), V = Object.keys(vocab).length, best = null, bestScore = -Infinity;
      Object.keys(classCounts).forEach(function (label) {
        var score = Math.log(classCounts[label] / docs);
        toks.forEach(function (t) { score += Math.log(((counts[label][t] || 0) + 1) / (totals[label] + V)); });
        if (score > bestScore) { bestScore = score; best = label; }
      });
      return best;
    }
    var api = { train: train, classify: classify, classes: function () { return Object.keys(classCounts); } };
    return api;
  };

  /* ------------------------------------------------- k-nearest neighbours */
  // "items like this one": returns a classifier fn.

  ML.knn = function (points, k) {
    return function (query) {
      var s = points.map(function (p) { return { label: p.label, d: euclid(query, p.x) }; });
      s.sort(function (a, b) { return a.d - b.d; });
      var tally = {};
      s.slice(0, k).forEach(function (p) { tally[p.label] = (tally[p.label] || 0) + 1; });
      var best = null, bc = -1;
      Object.keys(tally).forEach(function (l) { if (tally[l] > bc) { bc = tally[l]; best = l; } });
      return best;
    };
  };

  /* ------------------------------------------------------------- k-means */
  // Deterministic init (the first k points) so a run is reproducible.

  ML.kmeans = function (points, k, opts) {
    opts = opts || {};
    var maxIter = opts.maxIter || 100, dim = points[0].length, tol = opts.tol || 1e-9;
    var centroids = points.slice(0, k).map(function (p) { return p.slice(); });
    var labels = [], iter = 0, changed = true;
    while (changed && iter < maxIter) {
      changed = false; iter++;
      labels = points.map(function (p) {
        var bi = 0, bd = Infinity;
        centroids.forEach(function (c, i) { var d = euclid(p, c); if (d < bd) { bd = d; bi = i; } });
        return bi;
      });
      var sums = centroids.map(function () { return new Array(dim).fill(0); });
      var cnts = centroids.map(function () { return 0; });
      points.forEach(function (p, i) { var l = labels[i]; cnts[l]++; for (var j = 0; j < dim; j++) sums[l][j] += p[j]; });
      centroids = centroids.map(function (c, i) {
        if (cnts[i] === 0) return c;
        var nc = sums[i].map(function (s) { return s / cnts[i]; });
        if (euclid(nc, c) > tol) changed = true;
        return nc;
      });
    }
    return { centroids: centroids, labels: labels, iterations: iter };
  };

  /* ------------------------------------------------------------------ PCA */
  // Project to 2 dimensions for plotting (power iteration + deflation).

  ML.pca2 = function (points) {
    var n = points.length, dim = points[0].length;
    var mean = new Array(dim).fill(0);
    points.forEach(function (p) { for (var j = 0; j < dim; j++) mean[j] += p[j] / n; });
    var X = points.map(function (p) { return p.map(function (v, j) { return v - mean[j]; }); });
    var cov = [];
    for (var a = 0; a < dim; a++) cov[a] = new Array(dim).fill(0);
    X.forEach(function (x) { for (var i = 0; i < dim; i++) for (var j = 0; j < dim; j++) cov[i][j] += x[i] * x[j] / (n - 1); });
    function mv(M, v) { return M.map(function (row) { return row.reduce(function (s, val, j) { return s + val * v[j]; }, 0); }); }
    function norm(v) { return Math.sqrt(v.reduce(function (s, x) { return s + x * x; }, 0)); }
    function power(M) {
      var v = new Array(dim).fill(0).map(function (_, i) { return i === 0 ? 1 : 0.5; });
      for (var it = 0; it < 200; it++) { var nv = mv(M, v), nn = norm(nv) || 1; v = nv.map(function (x) { return x / nn; }); }
      return v;
    }
    var v1 = power(cov);
    var t1 = mv(cov, v1), l1 = v1.reduce(function (s, x, i) { return s + x * t1[i]; }, 0);
    var cov2 = cov.map(function (row, i) { return row.map(function (val, j) { return val - l1 * v1[i] * v1[j]; }); });
    var v2 = power(cov2);
    var comps = [v1, v2];
    var coords = X.map(function (x) { return comps.map(function (v) { return x.reduce(function (s, val, j) { return s + val * v[j]; }, 0); }); });
    return { points: coords, components: comps, mean: mean };
  };

  /* --------------------------------------------------- linear regression */
  // Ordinary least squares: a trend line.

  ML.linearRegression = function (xs, ys) {
    var n = xs.length, sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (var i = 0; i < n; i++) { sx += xs[i]; sy += ys[i]; sxx += xs[i] * xs[i]; sxy += xs[i] * ys[i]; }
    var denom = n * sxx - sx * sx;
    var slope = denom ? (n * sxy - sx * sy) / denom : 0;
    var intercept = (sy - slope * sx) / n;
    return { slope: slope, intercept: intercept, predict: function (x) { return slope * x + intercept; } };
  };

  /* ------------------------------------------------------- ID3 (tree) */

  function entropy(rows, target) {
    var counts = {}, n = rows.length || 1;
    rows.forEach(function (r) { counts[r[target]] = (counts[r[target]] || 0) + 1; });
    return -Object.keys(counts).reduce(function (s, k) { var p = counts[k] / n; return s + p * Math.log2(p); }, 0);
  }
  function majority(rows, target) {
    var counts = {}, best = null, bc = -1;
    rows.forEach(function (r) { counts[r[target]] = (counts[r[target]] || 0) + 1; });
    Object.keys(counts).forEach(function (k) { if (counts[k] > bc) { bc = counts[k]; best = k; } });
    return best;
  }

  ML.id3 = function (rows, attrs, target) {
    function build(rows, attrs) {
      var labels = {};
      rows.forEach(function (r) { labels[r[target]] = 1; });
      if (Object.keys(labels).length === 1) return { leaf: Object.keys(labels)[0] };
      if (attrs.length === 0 || rows.length === 0) return { leaf: majority(rows, target) };
      var base = entropy(rows, target), bestAttr = null, bestGain = -1;
      attrs.forEach(function (a) {
        var vals = {}, gain = base;
        rows.forEach(function (r) { vals[r[a]] = 1; });
        Object.keys(vals).forEach(function (v) {
          var sub = rows.filter(function (r) { return r[a] === v; });
          gain -= (sub.length / rows.length) * entropy(sub, target);
        });
        if (gain > bestGain) { bestGain = gain; bestAttr = a; }
      });
      var vset = {};
      rows.forEach(function (r) { vset[r[bestAttr]] = 1; });
      var children = {};
      Object.keys(vset).forEach(function (v) {
        var sub = rows.filter(function (r) { return r[bestAttr] === v; });
        children[v] = build(sub, attrs.filter(function (a) { return a !== bestAttr; }));
      });
      return { attr: bestAttr, children: children };
    }
    return build(rows, attrs.slice());
  };

  ML.predict = function (tree, row) {
    while (tree && tree.attr !== undefined) tree = tree.children[row[tree.attr]];
    return tree ? tree.leaf : undefined;
  };

  /* -------------------------------------------------------------- Markov */
  // Statistical generation from a corpus. char-level or word-level, n-gram order,
  // seedable so a generation is reproducible.

  ML.markov = function (text, opts) {
    opts = opts || {};
    var order = opts.order || 2;
    var word = !!opts.word;
    var tokens = word ? words(text) : String(text).split("");
    var table = {};
    for (var i = 0; i + order < tokens.length; i++) {
      var key = tokens.slice(i, i + order).join(word ? " " : "");
      (table[key] = table[key] || []).push(tokens[i + order]);
    }
    return { order: order, word: word, table: table };
  };

  ML.generate = function (model, opts) {
    opts = opts || {};
    var rng = ML.rng(opts.seed || 1), len = opts.length || 60;
    var keys = Object.keys(model.table);
    if (!keys.length) return "";
    var join = model.word ? " " : "";
    var key = (opts.start && model.table[opts.start]) ? opts.start : keys[Math.floor(rng() * keys.length)];
    var out = model.word ? key.split(" ") : key.split("");
    var cur = key;
    for (var i = 0; i < len; i++) {
      var nexts = model.table[cur];
      if (!nexts || !nexts.length) break;
      var nx = nexts[Math.floor(rng() * nexts.length)];
      out.push(nx);
      cur = (model.word ? cur.split(" ") : cur.split("")).concat(nx).slice(-model.order).join(join);
    }
    return out.join(join);
  };

  /* ------------------------------------------------------------- Grammar */
  // Rule-based generation: rules, alternatives, #slot# references, nested expansion,
  // weighted alternatives, seedable.

  ML.grammar = function (rules) {
    function pick(alts, rng) {
      var total = 0, list = alts.map(function (a) {
        var w = (typeof a === "string") ? 1 : (a.weight || 1);
        total += w; return { t: (typeof a === "string") ? a : a.text, w: w };
      });
      var r = rng() * total, acc = 0;
      for (var i = 0; i < list.length; i++) { acc += list[i].w; if (r < acc) return list[i].t; }
      return list[list.length - 1].t;
    }
    function expand(sym, rng, depth) {
      if (depth > 24) return "";
      var alts = rules[sym];
      if (!alts) return sym; // a literal, not a rule
      var chosen = pick(alts, rng);
      return chosen.replace(/#([a-zA-Z0-9_]+)#/g, function (_, s) { return expand(s, rng, depth + 1); });
    }
    return { expand: function (sym, seed) { return expand(sym, ML.rng(seed || 1), 0); } };
  };

  return ML;
}));
