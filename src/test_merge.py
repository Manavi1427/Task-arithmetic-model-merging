from merge import run_merge

run_merge(["sst2", "mrpc"], 0.5, "models/merged/test_run")
print("test merge done")