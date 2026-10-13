"""Path-configurable training, cross-validation and evaluation for EDFA models."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from data import load_csv
from models import MODEL_NAMES, build_model


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return result


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use --device cpu.")
    return torch.device(name)


def fit(x, y, args, device, seed):
    if len(x) < 2:
        raise ValueError("Training requires at least two rows for BatchNorm.")
    seed_all(seed)
    model = build_model(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    for epoch in range(args.epochs):
        model.train()
        batches = list(torch.randperm(len(x)).split(args.batch_size))
        # Merge a singleton remainder so BatchNorm can use every training row.
        if len(batches) > 1 and len(batches[-1]) == 1:
            batches[-2] = torch.cat((batches[-2], batches[-1]))
            batches.pop()
        squared_error, count = 0.0, 0
        for indices in batches:
            indices = indices.numpy()
            features = torch.from_numpy(x[indices]).to(device)
            target = torch.from_numpy(y[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features)
            mse = (prediction - target).square().mean()
            loss = torch.sqrt(mse + 1e-6)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss; inspect the data and learning rate.")
            loss.backward()
            optimizer.step()
            squared_error += float(mse.detach()) * target.numel()
            count += target.numel()
        rmse = (squared_error / count) ** 0.5
        history.append({"epoch": epoch + 1, "online_train_rmse": rmse})
        print(f"epoch {epoch + 1}/{args.epochs} online_train_rmse={rmse:.6f}", flush=True)
    return model, history


def evaluate(model, x, y, batch_size, device, predictions_path=None):
    model.eval()
    squared_error, absolute_error, count = 0.0, 0.0, 0
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            prediction = model(torch.from_numpy(x[start:start + batch_size]).to(device))
            if not torch.isfinite(prediction).all():
                raise ValueError("Model produced non-finite predictions.")
            prediction = prediction.cpu().double()
            target = torch.from_numpy(y[start:start + batch_size]).double()
            error = prediction - target
            squared_error += error.square().sum().item()
            absolute_error += error.abs().sum().item()
            count += error.numel()
            if predictions_path is not None:
                predictions.append(prediction.numpy())
    if predictions_path is not None:
        np.savetxt(predictions_path, np.concatenate(predictions), delimiter=",")
    return {
        "rows": len(x), "channels": 83, "elements": count,
        "squared_error_sum": squared_error, "absolute_error_sum": absolute_error,
        "rmse": (squared_error / count) ** 0.5, "mae": absolute_error / count,
        "units": "prepared CSV target units; no inverse transform applied",
    }


def save_run(output, model, history, args, data_hash, seed):
    configuration = {
        "model": args.model, "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.lr, "seed": seed, "data_sha256": data_hash,
        "input_features": 85, "output_features": 83, "preprocessing": "none",
        "optimizer": "Adam", "loss": "sqrt(MSE + 1e-6)", "scheduler": "none",
        "checkpoint_policy": "final epoch, no held-out checkpoint selection",
    }
    torch.save({"format_version": 1, "model_name": args.model,
                "state_dict": model.cpu().state_dict(), "configuration": configuration},
               output / "model.pt")
    write_json(output / "configuration.json", configuration)
    write_json(output / "history.json", history)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("train", "cross-validate", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True,
                             help="New output directory; existing directories are not overwritten.")
        command.add_argument("--skip-header", action="store_true")
        command.add_argument("--batch-size", type=positive_int, default=512)
        command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        if name == "evaluate":
            command.add_argument("--checkpoint", type=Path, required=True)
            command.add_argument("--save-predictions", action="store_true")
        else:
            command.add_argument("--model", choices=MODEL_NAMES, default="wide-cnn")
            command.add_argument("--epochs", type=positive_int, required=True)
            command.add_argument("--lr", type=float, default=0.001)
            command.add_argument("--seed", type=int, default=0)
        if name == "cross-validate":
            command.add_argument("--folds", type=positive_int, default=5)
    return result


def main():
    argument_parser = parser()
    args = argument_parser.parse_args()
    if args.command != "evaluate":
        if args.batch_size < 2 or not np.isfinite(args.lr) or args.lr <= 0:
            argument_parser.error("Training needs batch-size >= 2 and a finite positive learning rate.")
        if not 0 <= args.seed < 2**32 - getattr(args, "folds", 1):
            argument_parser.error("Seed is outside the supported range.")
    x, y = load_csv(args.data, args.skip_header)
    device = resolve_device(args.device)
    if args.command == "cross-validate":
        if not 2 <= args.folds <= len(x) or len(x) - int(np.ceil(len(x) / args.folds)) < 2:
            argument_parser.error("Each fold needs a nonempty test set and at least two training rows.")
    args.output.mkdir(parents=True, exist_ok=False)
    data_hash = digest(args.data)
    write_json(args.output / "environment.json", {
        "torch": str(torch.__version__), "numpy": np.__version__, "device": str(device),
        "data_sha256": data_hash, "skip_header": args.skip_header,
    })
    if args.command == "evaluate":
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 1:
            raise ValueError("Use a checkpoint produced by this runner, not an unlabelled legacy file.")
        model = build_model(checkpoint["model_name"])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(device)
        path = args.output / "predictions.csv" if args.save_predictions else None
        metrics = evaluate(model, x, y, args.batch_size, device, path)
        metrics["checkpoint_sha256"] = digest(args.checkpoint)
        write_json(args.output / "metrics.json", metrics)
        print(json.dumps(metrics, indent=2))
    elif args.command == "train":
        model, history = fit(x, y, args, device, args.seed)
        save_run(args.output, model, history, args, data_hash, args.seed)
    else:
        permutation = np.random.default_rng(args.seed).permutation(len(x))
        partitions = np.array_split(permutation, args.folds)
        metrics = []
        for fold, test_indices in enumerate(partitions):
            train_indices = np.concatenate([part for i, part in enumerate(partitions) if i != fold])
            output = args.output / f"fold-{fold + 1}"
            output.mkdir()
            print(f"fold {fold + 1}/{args.folds}", flush=True)
            # fit creates both a new model and a new optimizer for every fold.
            model, history = fit(x[train_indices], y[train_indices], args, device, args.seed + fold)
            result = evaluate(model, x[test_indices], y[test_indices], args.batch_size, device)
            result["fold"] = fold + 1
            write_json(output / "metrics.json", result)
            np.savez(output / "split.npz", train_indices=train_indices, test_indices=test_indices)
            save_run(output, model, history, args, data_hash, args.seed + fold)
            metrics.append(result)
        total = sum(item["elements"] for item in metrics)
        pooled_rmse = (sum(item["squared_error_sum"] for item in metrics) / total) ** 0.5
        write_json(args.output / "metrics.json", {"folds": metrics, "pooled_rmse": pooled_rmse})
        print(f"pooled held-out RMSE={pooled_rmse:.6f}")


if __name__ == "__main__":
    main()
