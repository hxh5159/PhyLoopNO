"""Source provenance independent of the original development Git history."""

def provenance(config, task=None):
    from cdlno.provenance import provenance as source_provenance
    return dict(source_provenance(task or config["task"]),
                code_version="phyloopno-paper-release")
