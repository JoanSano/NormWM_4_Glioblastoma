"""Quiet runs: stdout goes to a log file, and only a few lines reach the terminal."""

import contextlib
import os
import sys


class Tee:
    """Write to two streams at once, so a run is both shown and stored.

    Args:
        stream: The original stream, kept for `isatty` as well as for writing.
        handle: Open file the same output is mirrored into.
    """

    def __init__(self, stream, handle):
        """Store the `stream` and the `handle` the class docstring describes."""
        self.stream = stream
        self.handle = handle

    def write(self, data):
        """Write to both streams.

        Args:
            data: Text to write. Its length is returned, as `write` must.
        """
        self.stream.write(data)
        self.handle.write(data)
        return len(data)

    def flush(self):
        """Flush both streams. Takes no arguments."""
        self.stream.flush()
        self.handle.flush()

    def isatty(self):
        """Whether the original stream is a terminal. Takes no arguments."""
        return self.stream.isatty()


# Set by main(). Read by `announce`, which has to know whether a line it puts on
# the terminal is already going there through stdout.
ECHO_TO_TERMINAL = False


def announce(text):
    """Put one line on the terminal even while stdout is going to the log file.

    Args:
        text: The line to show.

    A run writes its output to the log rather than the screen, so the few lines
    that say what was produced and where have to bypass that redirection. Under
    --verbose stdout reaches the terminal anyway and this would print each of them
    twice, so it then only writes to the log.
    """
    print(text)
    if not ECHO_TO_TERMINAL:
        print(text, file=sys.__stdout__, flush=True)


@contextlib.contextmanager
def tee_stdout(path, echo=False):
    """Send everything printed to `path`, and to the terminal only if `echo`.

    Args:
        path: File the run is written to. Always a path, never None: it is the
            only record of the run, since the terminal no longer gets one.
        echo: Also keep writing to the terminal, as --verbose asks.

    Progress bars are unaffected either way: tqdm writes to stderr, which is not
    redirected, so a long run still shows that it is alive.
    """
    original = sys.stdout
    with open(path, "w") as handle:
        sys.stdout = Tee(original, handle) if echo else handle
        try:
            yield
        finally:
            sys.stdout = original


def resolve_log_path(log_arg, RESULTS):
    """The file the run is written to.

    Args:
        log_arg: The --log value: None or "" for the default name, otherwise the
            path asked for.
        RESULTS: Directory a relative path is resolved under.

    Never None. The terminal shows only what `announce` puts there, so a run that
    wrote no log would leave no record of itself at all.
    """
    path = log_arg or "createDatabase_log.txt"
    return path if os.path.isabs(path) else f"{RESULTS}/{path}"
