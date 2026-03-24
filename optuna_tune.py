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

    # Seed as a tunable categorical over fixed choices for fairer robustness
    config["seed"] = trial.suggest_categorical("seed", [42, 123, 999])
    config["deterministic"] = args.deterministic

    # Core optimization hyperparameters
    config["learning_rate"] = trial.suggest_float("learning_rate", 5e-5, 5e-3, log=True)
    config["weight_decay"] = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
    config["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64])
    config["hidden_dim"] = trial.suggest_categorical("hidden_dim", [32, 64, 96, 128])
    config["num_layers"] = trial.suggest_int("num_layers", 1, 3)
    config["dropout"] = trial.suggest_float("dropout", 0.0, 0.35)

    # Optional: teacher forcing schedule for sequence stability
    config["teacher_forcing_end"] = trial.suggest_float("teacher_forcing_end", 0.0, 0.4)

    # EVT-specific search only when using EVT loss
    if config.get("loss_type", "mse") == "evt_hybrid":
        config["evt_lambda"] = trial.suggest_float("evt_lambda", 0.005, 0.15, log=True)
        config["evt_xi"] = trial.suggest_float("evt_xi", 0.03, 0.30)
        config["evt_tail_quantile"] = trial.suggest_float("evt_tail_quantile", 0.85, 0.95)

        # Keep schedule/asymmetry as toggles to discover practical regimes
        use_schedule = trial.suggest_categorical("evt_use_lambda_schedule", [False, True])
        config["evt_use_lambda_schedule"] = use_schedule
        config["evt_asymmetric_penalty"] = trial.suggest_categorical("evt_asymmetric_penalty", [False, True])
        config["evt_under_penalty_multiplier"] = trial.suggest_float("evt_under_penalty_multiplier", 1.2, 3.0)

        if use_schedule:
            initial = trial.suggest_float("evt_lambda_initial", 0.005, 0.06, log=True)
            mid = trial.suggest_float("evt_lambda_mid", initial, 0.12)
            final = trial.suggest_float("evt_lambda_final", mid, 0.25)
            warmup = trial.suggest_int("evt_warmup_epochs", 8, max(10, args.epochs // 2))
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

        # Objective is minimum validation loss seen in this trial
        best_val_loss = metrics.get("best_val_loss", min(history["val_loss"]))
        return float(best_val_loss)

    return objective


def main():
    parser = argparse.ArgumentParser(description="Optuna tuning for GCN-LSTM")
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--epochs", type=int, default=35, help="Max epochs per trial")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience per trial")
    parser.add_argument("--study-name", type=str, default="gcnlstm_hpo")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_study.db")
    parser.add_argument("--timeout", type=int, default=None, help="Optional timeout in seconds")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic CUDNN mode")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel Optuna workers")
    args = parser.parse_args()

    # Make sure sqlite file directory exists when using sqlite storage
    if args.storage.startswith("sqlite:///"):
        db_path = args.storage.replace("sqlite:///", "", 1)
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

    sampler = optuna.samplers.TPESampler(multivariate=True, seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=6, interval_steps=1)

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
