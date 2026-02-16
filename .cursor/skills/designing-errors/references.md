# Error design reference (language-agnostic)

## Recommended error shape per layer

Use one structured type per library or service layer with these conceptual fields:

| Field | Purpose | Machine/Human |
|-------|---------|----------------|
| kind | Action-oriented category (NotFound, RateLimited, ValidationFailed, ...) | Machine |
| status | Permanent / Temporary / Persistent | Machine |
| message | Short, readable description | Both |
| context | List or map of (key, value): operation name, IDs, path, etc. | Human |
| source/cause | Underlying error for chaining (optional) | Human |

Kind and status drive retry and HTTP mapping; context and source drive debugging.

## Kind taxonomy (by action)

- **NotFound**: Resource missing; caller may create default or return 404.
- **PermissionDenied**: No access; do not retry.
- **RateLimited**: Back off and retry.
- **ValidationFailed**: Bad input; do not retry with same input.
- **Unavailable / Temporary**: Transient; retry with backoff.
- **Conflict**: State conflict; caller may resolve or return 409.

Adjust to domain; keep the question "what can the caller do?" in mind.

## Context at boundaries

At module or API boundaries, always add:

1. Operation name (e.g. "fetch_user", "save_document").
2. Relevant identifiers (user_id, task_id, path, request_id).

Pseudocode pattern:

```
function public_api(params) -> Result:
    result = internal_step(params)
    if result.is_err():
        return err with context:
            operation = "public_api"
            id = params.id
            source = result.err()
    return result
```

## Mapping to languages

- **Rust**: Flat struct with ErrorKind + ErrorStatus; use `.context()` or `or_raise()` at boundaries; consider exn-style trees for multi-cause.
- **Python**: Exception subclass with `kind`, `status`, `context` attributes; raise with context dict or helper that sets operation/ids.
- **TypeScript/JavaScript**: Error subclass or branded object with `kind`, `status`, `context`; wrap at layer boundaries with a small helper.
- **Go**: Custom type with Kind, Status, Context (map or struct); use `fmt.Errorf("op: %w", err)` plus set Kind/Status at boundaries.
- **Java/Kotlin**: Exception hierarchy or single type with kind/status/context; require context in factory methods for public API.

## Log format (human-oriented)

When logging, include:

1. Top-level message (what failed at this layer).
2. Location (file:line or span if cheap).
3. Context (operation, IDs).
4. Cause chain or tree (nested message/source), not only the root.

Example shape:

```
operation failed: save_document, doc_id=7829, at module.rs:45
  -> failed to fetch user "John Doe", at module.rs:52
  -> connection refused, at client.rs:89
```

## Machine handling (pseudocode)

```
match result
  Err(e) if e.kind == RateLimited and e.status == Temporary:
    sleep(backoff); retry()
  Err(e) if e.kind == NotFound:
    create_default()
  Err(e):
    return map_to_http(e.kind, e.status)
```

No chain traversal; one flat check on kind and status.
