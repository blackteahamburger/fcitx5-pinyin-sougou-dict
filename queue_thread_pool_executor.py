"""A thread pool executor using a queue for task management."""

import os
from concurrent.futures import Executor, Future
from queue import Empty, Queue
from threading import Condition, Thread
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


class QueueThreadPoolExecutor(Executor):
    """A thread pool executor using a queue for task management."""

    def __init__(self, max_workers: int | None = None) -> None:
        """
        Initialize a new QueueThreadPoolExecutor instance.

        Args:
            max_workers: The maximum number of threads that can be used to
                execute the given calls.

        Raises:
            ValueError: If max_workers is less than or equal to 0.
        """
        if max_workers is None:
            max_workers = min(32, (os.process_cpu_count() or 1) + 4)
        if max_workers <= 0:
            msg = "max_workers must be greater than 0"
            raise ValueError(msg)
        self._max_workers = max_workers
        self._task_queue: Queue[
            tuple[
                Callable[..., object],
                tuple[object, ...],
                dict[str, object],
                Future[Any],
            ]
            | None
        ] = Queue()
        self._shutdown = False
        self._task_condition = Condition()
        self._pending_tasks = 0
        for _ in range(self._max_workers):
            Thread(target=self._executor, daemon=True).start()

    def shutdown(  # ruff: ignore[complex-structure]
        self,
        wait: bool = True,  # ruff: ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument]
        *,
        cancel_futures: bool = False,
    ) -> None:
        """
        Clean-up the resources associated with the Executor.

        It is safe to call this method several times. Otherwise, no other
        methods can be called after this one.

        Args:
            wait: If True then shutdown will not return until all running
                futures have finished executing and the resources used by
                the executor have been reclaimed, and submissions remain
                open while outstanding tasks run, including submissions
                from worker tasks. If False then submissions close immediately.
            cancel_futures: If True then shutdown will cancel all pending
                futures. Futures that are completed or running will not be
                cancelled.
        """
        cancelled_futures: list[Future[object]] = []
        with self._task_condition:
            if cancel_futures:
                while True:
                    try:
                        task = self._task_queue.get_nowait()
                    except Empty:
                        break
                    if task is None:
                        self._task_queue.put(None)
                        self._task_queue.task_done()
                        break
                    _, _, _, future = task
                    cancelled_futures.append(future)
                    self._pending_tasks -= 1
                    self._task_queue.task_done()
                self._task_condition.notify_all()
            if not wait and not self._shutdown:
                self._shutdown = True
                for _ in range(self._max_workers):
                    self._task_queue.put(None)
        for future in cancelled_futures:
            future.cancel()
        if wait:
            with self._task_condition:
                while self._pending_tasks:
                    self._task_condition.wait()
                if not self._shutdown:
                    self._shutdown = True
                    for _ in range(self._max_workers):
                        self._task_queue.put(None)
            self._task_queue.join()

    def submit[**P, T](
        self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
    ) -> Future[T]:
        """
        Submits a callable to be executed with the given arguments.

        Schedules the callable to be executed as fn(*args, **kwargs) and
        returns a Future instance representing the execution of the
        callable.

        Args:
            fn: The function to execute.
            *args: Arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            A Future representing the given call.

        Raises:
            RuntimeError: If the executor is shut down.
                Note that this executor allows submissions after ``shutdown()`` has been
                called; a ``RuntimeError`` is raised only once the executor begins
                terminating threads and no longer accepts new tasks.
        """
        with self._task_condition:
            if self._shutdown:
                msg = "cannot schedule new futures after shutdown"
                raise RuntimeError(msg)
            future: Future[T] = Future()
            self._pending_tasks += 1
            self._task_queue.put((fn, args, kwargs, future))
        return future

    def _executor(self) -> None:
        while True:
            task = self._task_queue.get()
            if task is None:
                self._task_queue.task_done()
                return
            fn, args, kwargs, future = task
            try:
                if not future.set_running_or_notify_cancel():
                    continue
                result = fn(*args, **kwargs)
            except BaseException as exc:  # ruff: ignore[blind-except]
                future.set_exception(exc)
            else:
                future.set_result(result)
            finally:
                self._task_queue.task_done()
                with self._task_condition:
                    self._pending_tasks -= 1
                    self._task_condition.notify_all()
