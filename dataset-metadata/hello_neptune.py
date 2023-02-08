import neptune.new as neptune

# Create a Neptune run object
run = neptune.init_run(
    project="common/quickstarts",
    api_token=neptune.ANONYMOUS_API_TOKEN,
)

# Track metadata and hyperparameters by assigning them to the run
run["JIRA"] = "NPT-952"
run["algorithm"] = "ConvNet"

PARAMS = {
    "batch_size": 128,
    "dropout": 0.4,
    "learning_rate": 0.003,
    "optimizer": "Adam",
}
run["parameters"] = PARAMS

# Track the training process by logging your training metrics
for epoch in range(10):
    run["train/accuracy"].append(epoch * 0.6)
    run["train/loss"].append(epoch * 0.4)

# Record the final results
run["f1_score"] = 0.66

# Stop the connection and synchronize the data with the Neptune servers
run.stop()