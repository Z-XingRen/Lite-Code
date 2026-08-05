"""Native tool-definition accounting for context usage reports."""

import json


def native_tool_chars(agent):
    if hasattr(agent, "model_tools"):
        payload = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
                "strict": tool.strict,
            }
            for tool in agent.model_tools()
        ]
    else:
        payload = [
            {
                "name": name,
                "description": getattr(tool, "description", ""),
                "input_schema": getattr(
                    tool, "input_schema", getattr(tool, "schema", {})
                ),
                "strict": False,
            }
            for name, tool in agent.available_tools().items()
        ]
    return len(json.dumps(payload, ensure_ascii=False, sort_keys=True))
