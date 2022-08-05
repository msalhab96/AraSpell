import math
from typing import Any, Callable, Tuple, Union
TermCallback = Callable[
    [Union[float, int]], Tuple[bool, bool]
    ]


class TerminationCallback:
    def __init__(self, max_counts: int) -> None:
        self.max_counts = max_counts
        self.__counter = 0
        self.last_min = math.inf

    def __call__(
            self,
            last_val: Union[float, int],
            *args: Any, **kwargs: Any
            ) -> Tuple[bool, bool]:
        if last_val < self.last_min:
            self.last_min = last_val
            self.__counter = 0
            # Save checkpoint and don't terminate
            return True, False
        self.__counter += 1
        if self.__counter >= self.max_counts:
            # Don't Save checkpoint and terminate
            return False, True
        # Don't Save checkpoint and don't terminate
        return False, False


def get_callback(args) -> TermCallback:
    return TerminationCallback(args.stop_after)
