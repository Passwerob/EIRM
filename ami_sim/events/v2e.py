"""v2e subprocess adapter used for simulator ablations."""

from __future__ import annotations

from pathlib import Path

from .external import ExternalEventSimulatorBackend


class V2EBackend(ExternalEventSimulatorBackend):
    name = "v2e"

    def __init__(self, command: str | None = None, *, output_time_unit: str = "s", working_directory: str | Path | None = None) -> None:
        super().__init__(command, output_time_unit=output_time_unit, working_directory=working_directory)
