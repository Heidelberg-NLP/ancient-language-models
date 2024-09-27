log_file = "logs/lemmatization.log"

train_file = (
    "data/ud-treebanks-v2.14/UD_Ancient_Greek-Perseus/grc_perseus-ud-train.conllu"
)
valid_file = (
    "data/ud-treebanks-v2.14/UD_Ancient_Greek-Perseus/grc_perseus-ud-dev.conllu"
)
test_file = (
    "data/ud-treebanks-v2.14/UD_Ancient_Greek-Perseus/grc_perseus-ud-test.conllu"
)

output_dir = "models/lemmatization"


sweep_config = {
    "method": "grid",
    "parameters": {
        "learning_rate": {"values": [1e-5]},  # , 3e-5, 5e-5]
        "model_name_or_path": {"values": ["bowphs/GreTa"]},
        "num_train_epochs": {
            "values": [2],
        },
        "per_device_train_batch_size": {"values": [16]},
        "per_device_eval_batch_size": {"values": [16]},
        "weight_decay": {"values": [0.01]},
        "run_name": {"value": "demorun"},
        "predict_with_generate": {"value": True},
        "generation_max_length": {"value": 30},
        "generation_num_beams": {"value": 20},
    },
    "metric": {"name": "eval/exact_match", "goal": "maximize"},
}
