"""Coordinator subagent tool definitions."""

from ..core.worker_manager import dumps_payload

AGENT_TOOL_NAMES = {"agent", "send_message", "task_stop"}

AGENT_TOOL_SPECS = {
    "agent": {
        "risky": False,
        "description": "Launch a bounded worker or read-only Explore subagent.",
    },
    "send_message": {
        "risky": False,
        "description": "Continue an existing idle worker by id.",
    },
    "task_stop": {
        "risky": False,
        "description": "Stop a worker by id.",
    },
}

AGENT_TOOL_EXAMPLES = {
    "agent": {"description": "Inspect auth", "prompt": "Find auth entry points", "subagent_type": "Explore"},
    "send_message": {"to": "agent_1", "message": "Now patch the bug in src/auth.py"},
    "task_stop": {"task_id": "agent_1"},
}


def validate_agent_runtime(agent, name, args):
    """Runtime-aware checks that can't be expressed in the Pydantic schema."""
    if name == "agent":
        subagent_type = str(args.get("subagent_type", "worker")).strip()
        if agent.runtime_mode == "plan" and subagent_type != "Explore":
            raise ValueError("plan mode only allows Explore agents")


def tool_agent(agent, args):
    return dumps_payload(
        agent.worker_manager.spawn(
            args["description"],
            args["prompt"],
            subagent_type=args.get("subagent_type", "worker"),
            write_scope=args.get("write_scope", []),
        )
    )


def tool_send_message(agent, args):
    return dumps_payload(agent.worker_manager.continue_task(args["to"], args["message"]))


def tool_task_stop(agent, args):
    return dumps_payload(agent.worker_manager.stop_task(args["task_id"]))
