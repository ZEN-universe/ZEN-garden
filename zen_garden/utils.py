import os
import sys

class RedirectStdStreams(object):
    """
    A context manager that redirects the output to a file
    """

    def __init__(self, stdout=None, stderr=None):
        """
        Initializes the context manager
        :param stdout: Stream for stdout
        :param stderr: Stream for stderr
        """
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        """
        The exit function of the context manager
        :param exc_type: Type of the exit
        :param exc_value: Value of the exit
        :param traceback:  traceback of the error
        """
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr