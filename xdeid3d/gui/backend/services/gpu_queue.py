"""
GPU Queue Management - Prevents concurrent GPU operations
"""
import asyncio
from typing import Callable, Any
from dataclasses import dataclass
from enum import Enum


class JobType(Enum):
    SEED_GENERATION = "seed_generation"
    PTI_PROJECTION = "pti_projection"
    TEXT_TO_IMAGE = "text_to_image"


@dataclass
class Job:
    job_id: str
    job_type: JobType
    task: Callable
    priority: int = 0


class GPUQueue:
    """Queue for GPU operations to prevent OOM and conflicts"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.queue = asyncio.PriorityQueue()
        self.current_job = None
        self.is_processing = False

        self._initialized = True

    async def submit(self, job: Job) -> Any:
        """Submit a job to the queue"""
        await self.queue.put((job.priority, job))

        if not self.is_processing:
            asyncio.create_task(self._process_queue())

    async def _process_queue(self):
        """Process jobs from queue sequentially"""
        self.is_processing = True

        while not self.queue.empty():
            priority, job = await self.queue.get()
            self.current_job = job

            try:
                print(f"[GPUQueue] Processing job: {job.job_id} ({job.job_type.value})")
                result = await job.task()
                print(f"[GPUQueue] Completed job: {job.job_id}")
            except Exception as e:
                print(f"[GPUQueue] Error in job {job.job_id}: {e}")
            finally:
                self.current_job = None

        self.is_processing = False

    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self.queue.qsize()


# Global singleton instance
gpu_queue = GPUQueue()
