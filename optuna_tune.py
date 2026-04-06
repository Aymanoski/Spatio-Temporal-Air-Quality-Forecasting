"""
Optuna hyperparameter tuning for the GCN-LSTM model.

Usage examples:
  python optuna_tune.py --trials 30
  python optuna_tune.py --trials 50 --study-name bachelor-gcnlstm --storage sqlite:///optuna_study.db
"""

import argparse
import copy
import os

import optuna
from optuna.exceptions import TrialPruned

from train import CONFIG, train


def build_trial_config(trial, args):
    """Create a config dictionary for one Optuna trial."""
    config = copy.deepcopy(CONFIG)

    # Keep this as validation-focused tuning (no test leakage during study)
    config["evaluate_test"] = False
    config["save_checkpoints"] = False
    config["save_history"] = False
    config["resume"] = False
    config["epochs"] = args.epochs
    config["patience"] = args.patience
    config["architecture_name"] = f"optuna_trial_{trial.number}"

    # Keep trial noise low: use a fixed seed during search.
    # Robustness across multiple seeds should be checked after selecting best params.
    config["seed"] = args.seed
    config["deterministic"] = args.deterministic

    # Core optimization hyperparameters
    config["learning_rate"] = trial.suggest_float("learning_rate", 8e-5, 3e-3, log=True)
    config["weight_decay"] = trial.suggest_float("weight_decay", 1e-7, 5e-4, log=True)
    config["batch_size"] = trial.suggest_categorical("batch_size", [32, 64])
    config["hidden_dim"] = trial.suggest_categorical("hidden_dim", [64, 96])
    config["num_layers"] = trial.suggest_int("num_layers", 1, 2)
    config["dropout"] = trial.suggest_float("dropout", 0.0, 0.3)

    # Optional: teacher forcing schedule for sequence stability
    config["teacher_forcing_end"] = trial.suggest_float("teacher_forcing_end", 0.0, 0.25)

    # Tune wind-physics parameters for dynamic adjacency (single-stage search)
    if config.get("use_wind_adjacency", False):
        config["wind_alpha"] = trial.suggest_float("wind_alpha", 0.35, 0.9)
        config["distance_sigma"] = trial.suggest_float("distance_sigma", 1000.0, 4000.0, log=True)
        config["wind_recency_beta"] = trial.suggest_float("wind_recency_beta", 1.0, 5.0)
        config["wind_calm_speed_threshold"] = trial.suggest_float("wind_calm_speed_threshold", 0.02, 0.5)
        config["wind_aggregation_mode"] = trial.suggest_categorical(
            "wind_aggregation_mode", ["recent_weighted", "last", "mean"]
        )
        config["wind_direction_method"] = trial.suggest_categorical(
            "wind_direction_method", ["circular", "argmax_mean"]
        )
        config["wind_normalization"] = trial.suggest_categorical(
            "wind_normalization", ["row", "symmetric"]
        )

    # EVT-specific search only when using EVT loss
    if config.get("loss_type", "mse") == "evt_hybrid":
        config["evt_lambda"] = trial.suggest_float("evt_lambda", 0.008, 0.12, log=True)
        config["evt_xi"] = trial.suggest_float("evt_xi", 0.05, 0.25)
        config["evt_tail_quantile"] = trial.suggest_float("evt_tail_quantile", 0.85, 0.95)

        # Keep schedule/asymmetry as toggles to discover practical regimes
        use_schedule = trial.suggest_categorical("evt_use_lambda_schedule", [False, True])
        config["evt_use_lambda_schedule"] = use_schedule
        config["evt_asymmetric_penalty"] = trial.suggest_categorical("evt_asymmetric_penalty", [False, True])
        config["evt_under_penalty_multiplier"] = trial.suggest_float("evt_under_penalty_multiplier", 1.2, 3.0)

        if use_schedule:
            initial = trial.suggest_float("evt_lambda_initial", 0.008, 0.05, log=True)
            mid = trial.suggest_float("evt_lambda_mid", initial, 0.12)
            final = trial.suggest_float("evt_lambda_final", mid, 0.25)
            warmup = trial.suggest_int("evt_warmup_epochs", 8, max(10, args.epochs // 3))
            mid_epoch = trial.suggest_int("evt_mid_epochs", warmup + 1, max(warmup + 2, args.epochs - 1))
            config["evt_lambda_schedule"] = {
                "initial": initial,
                "mid": mid,
                "final": final,
                "warmup_epochs": warmup,
                "mid_epochs": mid_epoch,
                "transition": "smooth",
            }

    return config


def objective_factory(args):
    """Create objective with CLI arguments captured."""

    def objective(trial):
        config = build_trial_config(trial, args)
        try:
            _, history, metrics = train(config, trial=trial)
        except RuntimeError as exc:
            if str(exc) == "TRIAL_PRUNED":
                raise TrialPruned()
            raise

        # Objective is minimum val_mae (loss-function-independent, on original scale).
        # Using val_loss would cause Optuna to optimize the EVT loss value, not MAE,
        # and EVT lambda schedule changes would distort the search.
        best_val_mae = metrics.get("best_val_mae", min(history.get("val_mae", history["val_loss"])))
        return float(best_val_mae)

    return objective


def main():
    parser = argparse.ArgumentParser(description="Optuna tuning for GCN-LSTM")
    parser.add_argument("--trials", type=int, default=60, help="Number of Optuna trials")
    parser.add_argument("--epochs", type=int, default=45, help="Max epochs per trial")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience per trial")
    parser.add_argument("--study-name", type=str, default="gcnlstm_hpo")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_study.db")
    parser.add_argument("--timeout", type=int, default=None, help="Optional timeout in seconds")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic CUDNN mode")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel Optuna workers")
    parser.add_argument("--seed", type=int, default=42, help="Fixed seed used during the search")
    args = parser.parse_args()

    # Make sure sqlite file directory exists when using sqlite storage
    if args.storage.startswith("sqlite:///"):
        db_path = args.storage.replace("sqlite:///", "", 1)
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

    sampler = optuna.samplers.TPESampler(multivariate=True, seed=args.seed, n_startup_trials=12)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=12, n_warmup_steps=10, interval_steps=1)

    study = optuna.create_study(
        direction="minimize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    print("=" * 80)
    print("OPTUNA HYPERPARAMETER TUNING")
    print("=" * 80)
    print(f"Study name: {args.study_name}")
    print(f"Storage: {args.storage}")
    print(f"Trials: {args.trials}")
    print(f"Epochs/trial: {args.epochs}")
    print(f"Patience/trial: {args.patience}")
    print(f"Search seed: {args.seed}")
    print("Tune wind physics: True (always enabled)")
    print("=" * 80)

    study.optimize(
        objective_factory(args),
        n_trials=args.trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    print("\n" + "=" * 80)
    print("BEST TRIAL")
    print("=" * 80)
    print(f"Best value (val_loss): {study.best_value:.6f}")
    print("Best params:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    print("=" * 80)


if __name__ == "__main__":
    main()
