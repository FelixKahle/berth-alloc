# Chain-set: high-level overview

This module provides a compact state representation for multiple disjoint chains—ordered sequences of integer node IDs—with two goals in mind: fast local edits and safe, consistent reads during search. It is the low-level “sequence backbone” used by the solver to build and modify ordered plans, such as the sequence of vessels handled at a berth, while supporting speculative evaluation of moves before they are committed.

Each chain is bounded by two special sentinels that mark the start and end. An empty chain is simply “start followed by end.” Regular nodes that are not placed on any chain are self-looped, which makes it trivial to check if a node is still unused. This design allows the solver to insert, remove, and reorder nodes by rewiring local pointers, without scanning or reallocating larger structures.

A typical search step wants to try out a small change (for example, re-inserting a vessel earlier or later in a sequence), evaluate its impact on cost and feasibility, and then either commit or discard it. To make this cheap and safe, the module separates stable state from tentative edits. The stable state holds the canonical next/previous links for all chains. Tentative edits live in a lightweight overlay that records only what changed and presents a merged, read-only view to the rest of the solver. As a result, you can evaluate complex neighborhoods without touching the base state and apply accepted changes in one shot.

Because the representation is array-based and avoids pointer chasing through heap objects, iteration is cache-friendly and predictable. Local rewiring updates both directions consistently and isolates displaced nodes, so invariants are maintained even under heavy neighborhood exploration. Iteration over a chain never yields sentinels, and internal guards prevent accidental infinite walks when evaluating temporary configurations.

This module is intentionally minimal and generic. It does not impose domain semantics like time, capacity, or spatial feasibility; those are layered on top by the solver’s cost and constraint checks. The chain-set is about sequence structure only: fast, robust, and easy to reason about for heuristic and metaheuristic search.

In practice, you will:
- Keep a stable sequence state for the current solution, and stage candidate changes in a temporary overlay to score them before deciding to commit.
- Use the chain boundaries (start/end sentinels) as natural anchors for building empty chains, inserting nodes, and forming or breaking subsequences.

Glossary:
- Chain: an ordered sequence of node IDs representing, for example, the service order at a resource.
- Node: an integer index; unused nodes are self-looped.
- Sentinels: synthetic start/end markers for each chain that define empty chains and safe boundaries.

This structure underpins the solver’s ability to iterate quickly on sequence-based decisions, enabling large numbers of candidate evaluations per second while keeping the state consistent and the implementation easy to compose with higher-level logic.
