class DWILPScheduler:
    def __init__(self, config):
        self.config = config

    def schedule(self, tasks, nodes):
        raise NotImplementedError("Implement DW-ILP scheduler here.")
