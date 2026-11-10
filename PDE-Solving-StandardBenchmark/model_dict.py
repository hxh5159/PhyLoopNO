"""Registry for the paper model and its two parent baselines."""
from model import Transolver_Irregular_Mesh, Transolver_Structured_Mesh_2D


def get_model(args):
    if hasattr(args, '_linearno_loop_config'):
        from cdlno.linearno_loop.standard_entry import model_module
        return model_module(args)
    if args.model in ('LinearNO_Structured_Mesh_2D', 'LinearNO_Irregular_Mesh'):
        from model import LinearNO
        return LinearNO
    return {
        'Transolver_Irregular_Mesh': Transolver_Irregular_Mesh,
        'Transolver_Structured_Mesh_2D': Transolver_Structured_Mesh_2D,
    }[args.model]
