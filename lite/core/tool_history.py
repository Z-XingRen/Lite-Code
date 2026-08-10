"""Canonical persisted history item for one observed tool result."""


def build_tool_history_item(name, args, content, call_id, metadata, *, created_at):
    item = {
        "role": "tool",
        "name": name,
        "args": args,
        "content": content,
        "call_id": call_id,
        "created_at": created_at,
        "tool_status": str(metadata.get("tool_status", "")),
        "tool_error_code": str(metadata.get("tool_error_code", "")),
        "workspace_changed": bool(metadata.get("workspace_changed", False)),
        "affected_paths": list(metadata.get("affected_paths", []) or []),
    }
    if metadata.get("full_output_artifact"):
        item.update(
            {
                "artifact_ref": metadata["full_output_artifact"],
                "original_chars": int(metadata.get("original_chars", 0) or 0),
                "original_bytes": int(metadata.get("original_bytes", 0) or 0),
                "original_lines": int(metadata.get("original_lines", 0) or 0),
                "omitted_bytes": int(metadata.get("omitted_bytes", 0) or 0),
                "omitted_lines": int(metadata.get("omitted_lines", 0) or 0),
                "truncation_strategy": str(metadata.get("truncation_strategy", "")),
                "content_sha256": str(metadata.get("content_sha256", "")),
            }
        )
    if metadata.get("media_refs"):
        item["media_refs"] = list(metadata.get("media_refs", []) or [])
    return item
