import os
import sys
import errno
import shutil
import os.path as osp

import torch

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))


class Logger(object):
    """
    Write console output to an external text file in near real-time.
    """
    def __init__(self, fpath=None, buffer_size=1):
        """
        :param fpath: File path to the log file.
        :param buffer_size: Buffer size for writing to the file. 
                            1 indicates line buffering (write lines immediately).
                            Use a larger number for larger buffer size.
        """
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            # Open the file with buffering, 1 for line buffering.
            self.file = open(fpath, 'w', buffering=buffer_size)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)
            self.flush()  # Ensure the file is flushed after each write.

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            # os.fsync can be used here if immediate disk write is necessary,
            # but it may be commented out for better performance.
            # os.fsync(self.file.fileno())

    def close(self):
        # Do not close sys.stdout as it's used by the entire Python process.
        if self.file is not None:
            self.file.close()