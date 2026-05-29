import os
import json
import numpy as np
import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    args = parser.parse_args()

    exp_dir = args.exp_dir

    fold_files = [
        f for f in os.listdir(exp_dir)
        if f.startswith("fold") and f.endswith("_metrics.json")
    ]

    fold_files = sorted(fold_files)

    aucs = []
    accs = []
    sens = []
    specs = []
    y_sens = []
    y_specs = []
    epochs = []

    print("\n========== Fold Results ==========\n")

    for f in fold_files:

        path = os.path.join(exp_dir, f)

        with open(path) as j:
            data = json.load(j)

        fold = data["fold"]
        epoch = data.get("epoch", None)

        auc = data["auc"]
        acc = data["acc"]
        sensitivity = data["sensitivity"]
        specificity = data["specificity"]

        youden_sens = data.get("youden_sensitivity", None)
        youden_spec = data.get("youden_specificity", None)

        aucs.append(auc)
        accs.append(acc)
        sens.append(sensitivity)
        specs.append(specificity)

        if epoch is not None:
            epochs.append(epoch)

        if youden_sens is not None:
            y_sens.append(youden_sens)

        if youden_spec is not None:
            y_specs.append(youden_spec)

        print(
            f"Fold {fold}: "
            f"Epoch={epoch}, "
            f"AUC={auc:.4f}, "
            f"ACC={acc:.4f}, "
            f"SENS={sensitivity:.4f}, "
            f"SPEC={specificity:.4f}, "
            f"YOUDEN_SENS={youden_sens:.4f}, "
            f"YOUDEN_SPEC={youden_spec:.4f}"
        )

    print("\n========== Summary ==========\n")

    print(f"AUC         Mean ± Std: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"ACC         Mean ± Std: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"SENS        Mean ± Std: {np.mean(sens):.4f} ± {np.std(sens):.4f}")
    print(f"SPEC        Mean ± Std: {np.mean(specs):.4f} ± {np.std(specs):.4f}")

    if len(y_sens) > 0:
        print(f"YOUDEN_SENS Mean ± Std: {np.mean(y_sens):.4f} ± {np.std(y_sens):.4f}")

    if len(y_specs) > 0:
        print(f"YOUDEN_SPEC Mean ± Std: {np.mean(y_specs):.4f} ± {np.std(y_specs):.4f}")

    if len(epochs) > 0:
        print(f"BEST EPOCH  Mean ± Std: {np.mean(epochs):.2f} ± {np.std(epochs):.2f}")

    summary = {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),

        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),

        "sens_mean": float(np.mean(sens)),
        "sens_std": float(np.std(sens)),

        "spec_mean": float(np.mean(specs)),
        "spec_std": float(np.std(specs)),
    }

    if len(y_sens) > 0:
        summary["youden_sens_mean"] = float(np.mean(y_sens))
        summary["youden_sens_std"] = float(np.std(y_sens))

    if len(y_specs) > 0:
        summary["youden_spec_mean"] = float(np.mean(y_specs))
        summary["youden_spec_std"] = float(np.std(y_specs))

    if len(epochs) > 0:
        summary["epoch_mean"] = float(np.mean(epochs))
        summary["epoch_std"] = float(np.std(epochs))

    save_path = os.path.join(exp_dir, "cv_summary.json")

    with open(save_path, "w") as f:
        json.dump(summary, f, indent=4)

    print("\nSaved:", save_path)


if __name__ == "__main__":
    main()