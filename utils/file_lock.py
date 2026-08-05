"""
File-based lock to prevent concurrent writes to the same result file.
"""
import os, time, fcntl, random


class FileLock:
    """POSIX file lock with timeout."""

    def __init__(self, file_path, timeout=30):
        self.lock_path = file_path + '.lock'
        self.timeout = timeout
        self.fd = None

    def acquire(self):
        start = time.time()
        while True:
            try:
                self.fd = open(self.lock_path, 'w')
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.fd.write(str(os.getpid()))
                self.fd.flush()
                return True
            except (IOError, OSError):
                if self.fd:
                    self.fd.close()
                    self.fd = None
                if time.time() - start > self.timeout:
                    return False
                time.sleep(random.uniform(0.5, 2.0))

    def release(self):
        if self.fd:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            self.fd.close()
            self.fd = None
            try:
                os.remove(self.lock_path)
            except OSError:
                pass

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()


def safe_save(file_path, save_fn, timeout=60):
    """Execute save_fn(file_path) under a file lock."""
    lock = FileLock(file_path, timeout)
    if lock.acquire():
        try:
            save_fn(file_path)
        finally:
            lock.release()
