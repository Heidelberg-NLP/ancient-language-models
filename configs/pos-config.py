log_file = "logs/pos.log"

train_file = (
    "data/ud-treebanks-v2.14/UD_Ancient_Greek-Perseus/grc_perseus-ud-train.conllu"
)
valid_file = (
    "data/ud-treebanks-v2.14/UD_Ancient_Greek-Perseus/grc_perseus-ud-dev.conllu"
)
test_file = (
    "data/ud-treebanks-v2.14/UD_Ancient_Greek-Perseus/grc_perseus-ud-test.conllu"
)

output_dir = "models/pos"


sweep_config = {
    "method": "grid",
    "parameters": {
        "learning_rate": {"values": [1e-5]},  # , 3e-5, 5e-5]
        "model_name_or_path": {
            "values": [
                "bowphs/GreBerta"
            ]  # , 'bowphs/PhilBerta', 'pranaydeeps/Ancient-Greek-BERT']
        },
        "num_train_epochs": {
            "values": [2],
        },
        "per_device_train_batch_size": {"values": [16]},
        "per_device_eval_batch_size": {"values": [16]},
        "weight_decay": {"values": [0.01]},
        "run_name": {"value": "demorun"},
    },
    "metric": {"name": "eval/f1", "goal": "maximize"},
}
