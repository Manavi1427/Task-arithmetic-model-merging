from merge import task_config, run_merge

cfg = task_config(["sst2", "mrpc"], 0.5)
run_merge(cfg, "models/merged/test_run")
print("test merge done")

