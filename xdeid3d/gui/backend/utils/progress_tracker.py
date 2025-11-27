from typing import Callable, Optional, Any
import asyncio


class ProgressTracker:
    """Tracks progress and emits events"""

    def __init__(self, callback: Optional[Callable] = None):
        self.callback = callback
        self.current_stage = None
        self.current_step = 0
        self.total_steps = 0
        self.message = ""

    async def update(
        self,
        stage: str,
        step: int,
        total_steps: int,
        message: str = "",
        preview_image: Optional[str] = None,
    ):
        """Update progress and trigger callback"""
        self.current_stage = stage
        self.current_step = step
        self.total_steps = total_steps
        self.message = message

        if self.callback:
            progress = step / total_steps if total_steps > 0 else 0
            await self.callback({
                "type": "progress",
                "stage": stage,
                "progress": progress,
                "step": step,
                "total_steps": total_steps,
                "message": message,
                "preview_image": preview_image,
            })

    async def complete(self, result: Any):
        """Mark as complete and send result"""
        if self.callback:
            await self.callback({
                "type": "complete",
                "result": result,
            })

    async def error(self, error: str, code: str = "ERROR"):
        """Report error"""
        if self.callback:
            await self.callback({
                "type": "error",
                "error": error,
                "code": code,
            })
