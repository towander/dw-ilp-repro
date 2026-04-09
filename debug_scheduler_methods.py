import inspect
from src.learning_baselines.minimal_gnn_scheduler import GNNScheduler

print("GNNScheduler =", GNNScheduler)
print("init signature =", inspect.signature(GNNScheduler.__init__))
print("schedule signature =", inspect.signature(GNNScheduler.schedule))
