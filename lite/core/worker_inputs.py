"""Worker input normalization."""

from pathlib import Path


def clean_worker_type(value):
    subagent_type = str(value or "worker").strip()
    if subagent_type not in {"worker", "Explore"}:
        raise ValueError("subagent_type must be worker or Explore")
    return subagent_type


def clean_write_scope(value):
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise ValueError("write_scope must be a list of workspace paths")
    return [str(item).strip() for item in value if str(item).strip()]


def constrain_write_scope(runtime, requested_scope):
    """Narrow child scope to the parent runtime's write boundary."""

    requested = clean_write_scope(requested_scope)
    parent_scope = tuple(getattr(runtime, "write_scope", ()) or ())
    if not requested or not parent_scope:
        return requested

    effective = []
    for requested_item in requested:
        requested_path = runtime.path(requested_item)
        for parent_item in parent_scope:
            parent_path = runtime.path(parent_item)
            narrower = _narrower_path(requested_path, parent_path)
            if narrower is None:
                continue
            relative = Path(narrower).relative_to(runtime.root).as_posix()
            if relative not in effective:
                effective.append(relative)
    if not effective:
        raise ValueError("worker write_scope is outside the parent write_scope")
    return effective


def _narrower_path(left, right):
    try:
        left.relative_to(right)
        return left
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return right
    except ValueError:
        return None
