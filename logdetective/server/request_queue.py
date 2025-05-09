import asyncio
from typing import Callable

from logdetective.server.config import SERVER_CONFIG, LOG


async def process_queue(queue: asyncio.Queue):
    """Run an item on the queue every interval seconds.

    """
    while True:
        try:
            future, coro, args, kwargs = await queue.get()
            LOG.debug("run coroutine %s", coro)
            result = await coro(*args, **kwargs)
            LOG.debug("set result %s", result)
            future.set_result(result)
            queue.task_done()
        except asyncio.QueueEmpty:
            pass
        except asyncio.QueueShutDown:
            break

        await asyncio.sleep(SERVER_CONFIG.inference.request_period)


def enqueue_func(queue: asyncio.Queue, coro: Callable, *args, **kwargs):
    """ place coroutine in a queue and run it asynchronously """
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    LOG.debug("enque coroutine %s", coro)
    queue.put_nowait((future, coro, args, kwargs))
    return future
