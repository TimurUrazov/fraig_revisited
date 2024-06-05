import os
import json

from time import sleep
from typing import Any, Dict
from .common import date_now
from threading import Lock, Thread
import multiprocessing
import queue

from .work_path import WorkPath


class Logger:
    _name = None
    _session = None
    _lock = Lock()

    def __init__(self, out_path: WorkPath):
        self.out_path = out_path

    def __enter__(self):
        session = None
        path = str(self.out_path)
        name = f'{date_now()}_?'
        while session is None:
            try:
                os.mkdir(os.path.join(path, name))
                session = self.out_path.to_path(name)
            except FileExistsError:
                name = sleep(1) or f'{date_now()}_?'

        self._session = session
        self._name = name
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        path = self._session.base
        name = self._name.replace('?', sleep(1) or date_now())
        os.rename(path, path.replace(self._name, name))
        self._session, self._name = None, None

    def _write(self, string: str, filename: str) -> 'Logger':
        filepath = self._session.to_file(filename)
        with open(filepath, 'a+') as handle:
            handle.write(string)
        return self

    def format(self, obj: Dict[str, Any], filename: str) -> 'Logger':
        with self._lock:
            return self._write(f'{json.dumps(obj)}\n', filename)


class Log:
    def __init__(self, logger: Logger, logging_queue, filename: str):
        self._logger = logger
        self._filename = filename
        self.queue = logging_queue

        t = Thread(target=self._receive)
        t.daemon = True
        t.start()

    def _receive(self):
        with self._logger as logger:
            while True:
                try:
                    record = self.queue.get()
                    logger.format(record, self._filename)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except EOFError:
                    break

    def _send(self, s):
        self.queue.put_nowait(s)

    def emit(self, record):
        try:
            self._send(record)
        except (KeyboardInterrupt, SystemExit):
            raise


class MultiprocessingLog(Log):
    def __init__(self, logger: Logger, filename: str):
        super().__init__(logger, multiprocessing.Queue(-1), filename)


class MultithreadingLog(Log):
    def __init__(self, logger: Logger, filename: str):
        super().__init__(logger, queue.Queue(-1), filename)


__all__ = [
    'Logger',
    'MultiprocessingLog',
    'MultithreadingLog'
]
