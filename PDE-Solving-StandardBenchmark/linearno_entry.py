"""Pure LinearNO and PhyLoopNO lifecycle hooks for the six benchmarks."""
from cdlno.linearno.standard_entry import (
    model_kwargs, start, finish, normalizer, verify_data, StandardRun as PureStandardRun,
)


def StandardRun(args, model):
    if getattr(args, 'linearno_family', None) == 'linearno_loop':
        from cdlno.linearno_loop.standard_entry import LoopStandardRun
        return LoopStandardRun(args, model)
    return PureStandardRun(args, model)
