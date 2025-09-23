# berth-alloc

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Rust 2024](https://img.shields.io/badge/Rust-Edition%202024-orange.svg)](https://doc.rust-lang.org/edition-guide/rust-2024)
[![Test Status](https://img.shields.io/github/actions/workflow/status/FelixKahle/berth-alloc/test.yml?label=tests)](https://github.com/FelixKahle/berth-alloc/actions/workflows/test.yml)

---

A modular Rust workspace for experimenting with models and algorithms for the Berth Allocation Problem (BAP) and related quay scheduling problems. This is an academic/research codebase: the API and results are evolving, and breaking changes are likely as experiments progress.

---

## 🚢 Problem Description

The **Berth Allocation Problem (BAP)** is a classical NP-hard problem in maritime logistics.
It involves assigning vessels to berth positions over time, taking into account:

- Vessel dimensions and arrival windows
- Berth availability and compatibility
- Operational constraints (e.g., time windows, service durations)

Solving this problem efficiently is crucial for reducing congestion and improving
throughput at container terminals.

---

## Status

Actively evolving research code. Expect frequent changes and occasional breaking changes.

---

## Workspace layout

- `crates/berth-alloc-core`: shared utilities and common types
- `crates/berth-alloc-model`: BAP domain model (instances, constraints, objective components)
- `crates/berth-alloc-solver`: solver(s), heuristics/metaheuristics, and search primitives
- `crates/berth-alloc-main`: experiment runners

---

## Data and instances

The problem instances used in this project are provided by **Kramer–Lalla–Ruiz–Iori–Voss**: https://github.com/elalla/DBAP

---

## Design goals

- Separation of concerns between model and solver
- Reusable core utilities for experiments and benchmarking
- Extensible solver architecture (e.g., plug in heuristics or neighborhoods)

---

## License

This project is licensed under the MIT License. See [LICENSE.txt](LICENSE.txt) for details.
