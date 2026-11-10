"""Source fingerprints for the standalone PhyLoopNO project."""
def provenance(task=None):
    from cdlno.provenance import provenance as current
    return current(task)
