"""Runtime mode tool definitions."""

PLAN_TOOL_SPECS = {
    "enter_plan_mode": {
        "risky": False,
        "description": "Enter plan mode for a named planning topic.",
    },
    "exit_plan_mode": {
        "risky": False,
        "description": "Exit plan mode and return to default runtime mode.",
    },
}

PLAN_TOOL_EXAMPLES = {
    "enter_plan_mode": {"topic": "Refactor auth"},
    "exit_plan_mode": {},
}



def tool_enter_plan_mode(agent, args):
    path = agent.enter_plan_mode(args["topic"], path=args.get("path"))
    return f"mode: plan\nplan path: {path}"


def tool_exit_plan_mode(agent, args):
    agent.exit_plan_mode()
    return "mode: default"
