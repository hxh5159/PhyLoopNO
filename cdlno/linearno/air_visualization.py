"""LinearNO-only point sampling for hxh field plots; no graph-kernel dependency."""
import random
import torch
from cdlno.periodic_visualization import PeriodicFields


class AirFields(PeriodicFields):
    def _prepare(self, dataset, **kwargs):
        if not len(dataset):
            raise ValueError('visualization requires a nonempty held-out dataset')
        cases = []
        count = kwargs['hparams']['subsampling']
        for i in range(min(2, len(dataset))):
            data = dataset[i].clone().cpu()
            n = data.x.shape[0]
            if count > n:
                raise ValueError('AirfRANS visualization sample exceeds graph size')
            idx = torch.tensor(random.Random(i).sample(range(n), count))
            for field in ('pos', 'x', 'y', 'surf'):
                setattr(data, field, getattr(data, field)[idx].clone())
            data.edge_index = None
            cases.append((data, idx, n))
        return cases
