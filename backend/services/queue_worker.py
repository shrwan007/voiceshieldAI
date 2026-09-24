"""
Async Processing Queue
Queues large audio file uploads for background processing.
Prevents blocking the API for long-running inference tasks.
"""

import asyncio
import uuid
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ProcessingJob:
    job_id: str
    file_bytes: bytes
    filename: str
    user_id: Optional[str]
    session_id: str
    status: JobStatus = JobStatus.PENDING
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    result: Optional[dict] = None
    error: Optional[str] = None

class ProcessingQueue:
    """
    FIFO async queue for audio processing jobs.
    A background worker processes jobs as they arrive.
    """
    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._jobs: dict[str, ProcessingJob] = {}
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
    
    async def enqueue(
        self,
        file_bytes: bytes,
        filename: str,
        user_id: Optional[str] = None
    ) -> str:
        """
        Add a file to the processing queue.
        Returns job_id for status polling.
        Raises QueueFull if queue is at capacity.
        """
        job_id = str(uuid.uuid4())
        session_id = f"job_session_{job_id}"
        job = ProcessingJob(
            job_id=job_id,
            file_bytes=file_bytes,
            filename=filename,
            user_id=user_id,
            session_id=session_id
        )
        self._jobs[job_id] = job
        
        try:
            self._queue.put_nowait(job_id)
        except asyncio.QueueFull:
            self._jobs.pop(job_id)
            raise Exception("Queue is full")
            
        return job_id
    
    async def process_jobs(self) -> None:
        """
        Background worker — runs indefinitely processing jobs from queue.
        Call this in a background task on app startup.
        """
        # Note: avoid circular import here by importing when needed or ensuring safe pipeline import
        from backend.pipeline import run_full_pipeline
        
        while self._running:
            try:
                job_id = await self._queue.get()
                job = self._jobs.get(job_id)
                if not job:
                    self._queue.task_done()
                    continue
                
                job.status = JobStatus.PROCESSING
                
                try:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, 
                        run_full_pipeline, 
                        job.file_bytes, 
                        job.filename, 
                        job.user_id,
                        None 
                    )
                    
                    job.result = result
                    job.status = JobStatus.COMPLETED
                except Exception as e:
                    job.error = str(e)
                    job.status = JobStatus.FAILED
                finally:
                    job.completed_at = time.time()
                    self._queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in queue worker: {e}")
                await asyncio.sleep(1)
    
    async def get_job_status(self, job_id: str) -> dict:
        """Return current status of a job by ID."""
        job = self._jobs.get(job_id)
        if not job:
            return {"error": "Job not found"}
        
        return {
            "job_id": job.job_id,
            "status": job.status,
            "created_at": job.created_at,
            "completed_at": job.completed_at,
            "result": job.result,
            "error": job.error
        }
    
    async def start(self) -> None:
        """Start the background worker task."""
        if not self._running:
            self._running = True
            self._worker_task = asyncio.create_task(self.process_jobs())
    
    async def stop(self) -> None:
        """Gracefully stop the worker."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

# Module-level singleton
worker_queue = ProcessingQueue()
