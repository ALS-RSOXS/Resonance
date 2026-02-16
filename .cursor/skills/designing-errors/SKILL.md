---
name: designing-errors
description: Designs and formats errors for machines and humans instead of forwarding them. Use when defining error types, handling failures, adding context to errors, or when the user mentions error handling, error design, debugging production failures, retry logic, or error messages.
---

# Designing Errors (Not Forwarding Them)

Errors are messages for two audiences. Optimize for both, not for the compiler or "getting types to line up."

## Two Audiences

| Audience | Goal | Needs |
|----------|------|--------|
| Machines | Automated recovery | Flat structure, clear kinds, predictable codes, explicit retryability |
| Humans | Debugging | Rich context, logical path (what was being done), business-level identifiers |

## Core Principles

### 1. Categorize by action, not origin

Define error kinds by **what the caller can do**, not which dependency failed.

- Good: `NotFound`, `RateLimited`, `PermissionDenied`, `ValidationFailed`
- Avoid: `DatabaseError`, `HttpError`, `SerializationError` (origin-only)

When handling an error, the caller should know: retry? create default? return 404? The kind should answer that.

### 2. Make retryability explicit

Do not infer retryability from the error type. Expose it as a first-class field or attribute.

- Permanent: do not retry
- Temporary: safe to retry (e.g. rate limit, timeout)
- Persistent: was retried and still failing (optional, for observability)

Machines can then branch on status without traversing chains or guessing.

### 3. Add context at every layer

Every place that catches or wraps an error should add one line of context: what was being attempted (operation name, IDs, request path). If adding context is high-friction, it will be skipped.

- Enforce context at module or API boundaries (by type, by convention, or by required parameters).
- Prefer a single structured error type per layer with a context/message field rather than a new variant per call site.

### 4. Design for the 3am log line

Before returning or rethrowing, ask: "If this fails in production, what would I wish the log said?" Then make the error carry that: operation, identifiers (user id, task id, path), and the underlying cause. Preserve a logical path (e.g. tree or chain) so humans see the flow, not only the root exception.

### 5. Prefer one flat error type per layer

One error type per library or service layer, with:

- kind (action-oriented)
- status (retryability)
- message
- context (key-value or list of strings for operation, IDs, etc.)
- optional source/cause for chaining

Avoid scattering many small error enums; use context for specificity.

## Checklist when defining or handling errors

- [ ] Error kind answers "what can the caller do?" not only "where did it come from?"
- [ ] Retryability is explicit (permanent / temporary), not implied
- [ ] Every boundary adds context (operation + relevant IDs/path)
- [ ] Log/output would show logical path and business identifiers, not only the root message
- [ ] One structured error type per layer; context field for detail

## Anti-patterns

- **Forwarding only**: Catch, wrap with no new context, rethrow. The message is preserved but meaning is lost.
- **Origin-only kinds**: Error types that only name the dependency (DB, HTTP, JSON) and give no hint for recovery.
- **Optional context**: Context is possible but not required; developers skip it under pressure.
- **Expensive backtraces as the only debug info**: Prefer cheap, explicit context (location, operation, IDs) where the language supports it.

## Additional resources

- For concrete shapes and multi-language patterns, see [reference.md](reference.md).
