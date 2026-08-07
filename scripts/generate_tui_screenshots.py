from __future__ import annotations

import asyncio
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lite import Lite, SessionStore, WorkspaceContext  # noqa: E402
from lite.testing import ScriptedModelClient  # noqa: E402
from lite.tui.app import LiteTuiApp  # noqa: E402
from lite.tui.widgets import InputBar  # noqa: E402


OUTPUT_DIR = ROOT / "assets" / "screenshots"
SCREEN_SIZE = (120, 34)


async def _wait_until_ready(bar: InputBar, pilot, attempts: int = 60) -> None:
    for _ in range(attempts):
        await pilot.pause(delay=0.05)
        if not bar.input.disabled:
            return
    raise RuntimeError("timed out waiting for the TUI input")


async def _capture(
    filename: str,
    *,
    command: str | None = None,
    outputs: list[str] | None = None,
    leave_input: bool = False,
) -> None:
    with TemporaryDirectory(prefix="lite-tui-") as temp_dir:
        workspace_root = Path(temp_dir)
        (workspace_root / "README.md").write_text("# Demo repository\n", encoding="utf-8")
        client = ScriptedModelClient(outputs or [])
        client.model = "gpt-5.4"
        client.reasoning_effort = "high"
        agent = Lite(
            model_client=client,
            workspace=WorkspaceContext.build(workspace_root),
            session_store=SessionStore(workspace_root / ".lite" / "sessions"),
            approval_policy="auto",
        )
        app = LiteTuiApp(agent)
        async with app.run_test(size=SCREEN_SIZE) as pilot:
            await pilot.pause()
            if command is not None:
                bar = app.query_one(InputBar)
                bar.input.value = command
                if leave_input:
                    bar.update_slash_suggestions()
                    await pilot.pause()
                else:
                    await pilot.press("enter")
                    await _wait_until_ready(bar, pilot)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUTPUT_DIR / filename).write_text(
                app.export_screenshot(title="LITE Code"),
                encoding="utf-8",
            )


async def main() -> None:
    await _capture("lite-tui-intro.svg")
    await _capture(
        "lite-tui-tools.svg",
        command="Create notes/result.txt with a short status update.",
        outputs=[
            '<tool name="write_file" path="notes/result.txt"><content>ready\n</content></tool>',
            "<final>Created the status note.</final>",
        ],
    )
    await _capture("lite-tui-skills-help.svg", command="/help")
    await _capture("lite-tui-memory-skills.svg", command="/memory")
    await _capture("lite-tui-latest.svg", command="/", leave_input=True)


if __name__ == "__main__":
    asyncio.run(main())
