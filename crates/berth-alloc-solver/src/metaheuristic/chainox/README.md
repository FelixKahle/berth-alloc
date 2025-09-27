# Chainox: In-Place Path Arena for Metaheuristics

Chainox is a minimal, high-performance path arena for metaheuristic search. It enables extremely fast manipulation of solution sequences by avoiding heap allocations and complex ownership.

It models multiple disjoint, doubly-linked paths that can be rewired in constant time, making it the perfect backbone for local search and large-neighborhood search algorithms.

---
## Core Features

* **⚡️ Constant-Time Splicing:** Rewire and move entire sub-sequences in `O(1)` time, perfectly mirroring common neighborhood moves.

* **♻️ Allocation-Free Edits:** All path manipulations—insertions, removals, and moves—are performed in-place without heap churn, making inner loops incredibly fast.

* **🧠 Cache-Friendly Design:** A flat, data-oriented layout ensures that iterating and updating paths is highly efficient.

* **🔗 Decoupled Data:** The arena only manages the sequence topology. Your domain data remains separate, allowing for a clean and safe integration.

* **🛡️ Safe & Debuggable:** A clear API with strong error types prevents illegal moves, while debug-only invariant checks help catch logic errors early.
