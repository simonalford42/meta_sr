"""
Utility functions for meta-evolution
"""
import signal
import time
import traceback


class TimeoutError(Exception):
    """Raised when execution times out"""
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")


def run_with_timeout(func, args=(), kwargs=None, name="function",
                     soft_timeout=0.326, hard_timeout=1):
    """
    Run a function with soft and hard timeouts.

    Args:
        func: The function to run
        args: Positional arguments for func
        kwargs: Keyword arguments for func
        name: Name for error messages (e.g., "selection operator", "test 'random_fitness'")
        soft_timeout: Soft timeout in seconds (returns failure if exceeded, but only checked after completion)
        hard_timeout: Hard timeout in seconds (kills execution via signal, catches infinite loops)

    Returns:
        (success, elapsed_time, error_message)
        - success: True if completed within soft_timeout, False otherwise
        - elapsed_time: Time taken in seconds (None if hard timeout hit)
        - error_message: None if success, otherwise description of failure
    """
    if kwargs is None:
        kwargs = {}

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)

    try:
        signal.alarm(hard_timeout)

        start = time.perf_counter()
        func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        signal.alarm(0)

        if elapsed > soft_timeout:
            return False, elapsed, f"{name} timed out: {elapsed:.4f}s > {soft_timeout:.4f}s (soft timeout)"

        return True, elapsed, None

    except TimeoutError:
        signal.alarm(0)
        return False, None, f"{name} killed: infinite loop or exceeded {hard_timeout}s hard timeout"

    except Exception as e:
        signal.alarm(0)
        return False, None, (e, f"{name} failed: {e}")

    finally:
        signal.signal(signal.SIGALRM, old_handler)


def _find_lineno_in_generated_code(exc):
    """
    Find the line number in generated code (<string>) from an exception traceback.

    Walks the traceback to find the last frame that's in <string> (the exec'd code),
    rather than the innermost frame which might be in library code.
    """
    if exc is None or exc.__traceback__ is None:
        return None

    error_lineno = None
    tb = exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == "<string>":
            error_lineno = tb.tb_lineno
        tb = tb.tb_next

    return error_lineno


def print_code_with_error(code, exc=None, title="Generated code"):
    """
    Print code with the error line highlighted (if exception provided).

    Args:
        code: The source code string
        exc: Optional exception with traceback to highlight the error line
        title: Title for the code block
    """
    error_lineno = _find_lineno_in_generated_code(exc)

    if error_lineno is not None:
        print(f"\n--- {title} with error ---")
        for i, line in enumerate(code.split('\n'), 1):
            marker = "-->" if i == error_lineno else "   "
            print(f"{marker} {i:3d} | {line}")
        print("-" * (len(title) + 20) + "\n")
    else:
        print(f"\n--- {title} ---")
        for i, line in enumerate(code.split('\n'), 1):
            print(f"    {i:3d} | {line}")
        print("-" * (len(title) + 8) + "\n")


def print_error_in_generated_code(code, exc, name="function"):
    """
    Print full error information for LLM-generated code.

    Args:
        code: The source code string
        exc: The exception that was raised
        name: Name of what failed (for error message)
    """
    print(f"{name} failed: {exc}")
    traceback.print_exception(type(exc), exc, exc.__traceback__)
    print_code_with_error(code, exc, title="Generated code")
