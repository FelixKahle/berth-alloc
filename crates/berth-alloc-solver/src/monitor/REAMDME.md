# Monitor subsystem

⏱️ The monitor keeps your search loop honest, fast, and stoppable. It centralizes stopping criteria, budgets, and progress observation without getting in the way of performance.

The system lives in the `monitor` module and is made up of a few small pieces that snap together:
- `controller`: the brain that owns configuration (`SearchLimits`), a global `StopToken`, and improvement bookkeeping.
- `lease`: fast budget accounting with a global `ChunkDispenser` and per-thread `Lease`s.
- `termination`: a thread-local helper that you “tick” in hot loops to consume budgets and periodically check global limits.
- `stop`: a tiny `StopToken` and an `ImprovementState` for “no improvement” logic.
- `observer`: lifecycle hooks you can plug in for logs, metrics, or UI without touching core search logic.

Why this exists
Search loops often need to stop based on time, iteration counts, or when there’s no improvement. Doing this naïvely can hammer atomics, clutter inner loops, and make multi-threaded coordination hard. The monitor system keeps the hot path unbelievably light, while still letting you enforce global limits across all threads.

How it works at a glance

1. You choose your limits in `SearchLimits`: maximum duration, maximum neighbors or iterations (global), and optional “no-improvement” thresholds (time and/or accepted count). There are also two small performance knobs: `neighbor_chunk` and `iteration_chunk` control how many tokens are leased per thread at a time, and `sample_every` decides how often the slow checks run.

2. You instantiate a shared `GlobalController`. It owns a `StopToken`, tracks elapsed time, and maintains an `ImprovementState`. If you configured budgets, it builds a `ChunkDispenser` for each one.

3. Each worker thread constructs a local `Termination` from the controller. That object holds optional `Lease`s for budgets and a copy of the `StopToken`. On each outer iteration, the thread calls `tick_iteration()`. When generating neighbors, it calls `tick_neighbor()`. These methods decrement a fast, thread-local counter most of the time, and only touch a single relaxed atomic when a chunk runs dry. Every `sample_every` calls, the same methods also check the “slow” global conditions (time budget, no-improve thresholds, or external stop).

4. When the global budget is exhausted, the dispenser sets the shared `StopToken`. All threads will see it shortly via their next tick or via the super-cheap `should_stop_fast()` check.

5. When a plan is accepted or the incumbent improves, you notify the controller with `on_accepted()` or `on_incumbent_improvement()`. That’s how the “no-improve” thresholds are enforced without polluting operator code.

6. If you want progress reporting, you can attach `SearchObserver`s (for example, a composite observer that forwards events to logging, metrics, and a status UI). It’s optional and separate from control logic, so your hot loops stay clean.

Why it’s fast

🧮 Chunked budgets mean the hot path only does simple integer math most of the time. Threads get a handful of “tokens” up front via a `Lease`; decrementing a local `u64` has no contention. Only when the lease empties do you touch a relaxed atomic to refill (and the very last refill clamps to the exact global limit).

⏱️ Sampled checks keep the slow stuff out of the inner loop. `Termination` runs the more expensive checks (time, “no improvement”) every `sample_every` ticks, which you can tune. Between samples, it’s only a couple of arithmetic operations and a branch.

🚦 The stop signal is cooperative and cheap. A single relaxed `AtomicBool` (`StopToken`) tells everyone to wrap up. You can poll it directly in ultra-tight places with `should_stop_fast()`.

What you get out of the box

- Absolute time budgets via `max_duration`.
- Global counts for neighbors and/or iterations, shared across threads, enforced without overconsumption.
- “No-improvement” cutoffs by elapsed time since last improvement or by number of accepted plans since then.
- Thread-local helpers that you can “tick” anywhere in your loop without dragging in controller state.
- Optional observers for start/end, accepted iterations, and new incumbents, with a simple composite pattern.
- Human-friendly `Display` outputs for limits, leases, controller, and termination for quick logging.

Typical flow in prose

You configure `SearchLimits`, spin up a shared `GlobalController`, and then start your search threads. Each thread wraps the controller in a `Termination` and enters the main loop. On each iteration, it calls `tick_iteration()`; while enumerating neighbors, it calls `tick_neighbor()` every time it produces one. The thread evaluates neighbors, and if a plan is accepted, it pings `on_accepted()`; if that plan improves the incumbent, it pings `on_incumbent_improvement()`. From time to time, the `Termination` samples the global state; if time’s up, budgets are exhausted, or the no-improvement threshold hits, the shared `StopToken` flips. All threads observe the stop and wind down gracefully.

Tuning tips

🔧 For extremely cheap neighbors, increase `neighbor_chunk` and `sample_every` to push contention lower and sample less often. For tighter control or responsiveness to time/no-improve limits, decrease `sample_every`. For fairness across many threads, you may want smaller chunks; for raw throughput, larger chunks reduce refills.

Threading model

🧵 The controller is shared (usually wrapped in `Arc`). Each thread owns its `Termination` and optional `Lease`s. The only shared atomics are in the dispensers (on refill) and the `StopToken` (set once and read cheaply). Budgets are exact: the final lease grant is truncated so the total tokens consumed never exceed the configured limit.

Observability

📊 You can observe start/end events, accepted iterations (with old/new cost), and new incumbents via `SearchObserver`. A `CompositeSearchObserver` lets you fan out to multiple consumers. This is optional: your control logic remains independent.

Key types to know by name

- `SearchLimits` and `GlobalController` (in `controller`)
- `ChunkDispenser` and `Lease` (in `lease`)
- `Termination` (in `termination`)
- `StopToken` and `ImprovementState` (in `stop`)
- `SearchObserver`, `NullObserver`, `CompositeSearchObserver` (in `observer`)

In short

🧠 The monitor subsystem enforces when to stop, with minimal overhead, across any number of threads. It separates policy (limits and improvement semantics) from mechanism (fast ticking and cooperative stop) and keeps your operators and loops tidy, testable, and performant.
