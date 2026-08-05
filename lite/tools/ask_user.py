"""User clarification tool definitions."""

ASK_USER_TOOL_SPECS = {
    "ask_user": {
        "risky": False,
        "description": "Ask the interactive user a real blocking clarification question.",
    },
}

ASK_USER_TOOL_EXAMPLES = {
    "ask_user": {"question": "Which target should I deploy?", "choices": ["staging", "production"]},
}



def tool_ask_user(agent, args):
    return agent.ask_user(str(args["question"]), choices=args.get("choices", []) or [])
