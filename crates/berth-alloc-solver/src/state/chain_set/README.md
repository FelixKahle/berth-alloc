Chain-set: fast, safe sequences for search 🚀

Chain-set is a compact data structure for managing multiple disjoint chains—ordered sequences of integer node IDs. It’s the solver’s low-level “sequence backbone,” built for two things:
	•	Fast local edits (insert, remove, reorder) in O(1) time per edge change.
	•	Safe, consistent reads while you explore moves during search.

It keeps the stable state separate from tentative edits, so you can try complex neighborhoods cheaply, score them, and then commit only the winners.

⸻

Why this exists 💡

Heuristic and metaheuristic search spend most of their time nudging sequences: “what if we move this vessel earlier?”, “what if we splice this block over there?”. Chain-set lets you do that with tiny, local rewires—no scans, no allocations—while a read-only overlay provides a merged view for evaluation.

⸻

Core ideas (at a glance)
	•	Sentinel-bounded chains: each chain has a start and end sentinel.
	•	Empty chain = start → end.
	•	Iteration never yields sentinels.
	•	Unused nodes self-loop: next[n] = prev[n] = n.
Quick check for “not placed yet.”
	•	Local rewiring updates both directions and isolates displaced nodes.
	•	Overlay for tentative edits: record only changed edges; everyone else reads a merged view.
Score first, apply once if accepted.

⸻

Typical workflow 🛠️
	1.	Keep the current solution in the base chain-set.
	2.	Build a delta/overlay with a few set_next rewires (or use the builder helpers).
	3.	Evaluate costs/feasibility against the overlay’s read-only view.
	4.	If it helps, apply the delta to the base in one shot; otherwise discard.

⸻

What stays safe ✅
	•	No edge ever points to a head sentinel.
	•	You never modify a tail sentinel as a tail.
	•	The exact edge you touch is kept locally consistent (next[tail] == head, prev[head] == tail).
	•	Iterators guard against accidental infinite walks (bounded steps).

Global shape (unique membership, full head→…→tail connectivity) may be temporarily broken while you’re composing moves—that’s intentional for speed. Use the overlay to evaluate, then finalize to restore global structure.

⸻

Performance notes ⚡
	•	Array-based (cache-friendly) next/prev links; no pointer chasing through heap objects.
	•	Edits are O(1) per rewire; applying a delta is linear in the number of touched edges.
	•	Plays nicely with higher-level cost/constraint layers (time, capacity, spatial rules), which remain orthogonal to sequence structure.

⸻

Glossary 📚
	•	Chain: ordered sequence of node IDs (e.g., service order at a berth).
	•	Node: integer index; unused nodes are self-looped.
	•	Sentinels: per-chain start/end markers that define boundaries and empty chains.
