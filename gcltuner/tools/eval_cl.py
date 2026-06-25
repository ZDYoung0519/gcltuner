import argparse
import os
import os.path as osp

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True)
    return parser.parse_args()


def read_metric(metric_file: str) -> float:
    with open(metric_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    metric = lines[-1].strip().split(":")[1].replace("%", "")
    dataset_name = lines[0].strip().split(":")[1]
    return dataset_name, np.round(float(metric), 3)


def task_id(name: str):
    if name.startswith("task") and name[4:].isdigit():
        return int(name[4:])
    return None


def main():
    args = parse_args()

    task_dirs = [
        name for name in os.listdir(args.eval_dir)
        if task_id(name) is not None and osp.isdir(osp.join(args.eval_dir, name))
    ]

    n = max(task_id(name) for name in task_dirs) + 1
    matrix = np.full((n, n), np.nan)

    task_names = {}
    for train_name in task_dirs:
        i = task_id(train_name)
        train_dir = osp.join(args.eval_dir, train_name)

        for eval_name in os.listdir(train_dir):
            j = task_id(eval_name)
            if j is None:
                continue

            metric_file = osp.join(train_dir, eval_name, "metric.txt")
            if osp.isfile(metric_file):
                task_name, matrix[i, j] = read_metric(metric_file)
                task_names[j] = task_name

    labels = [task_names[i] for i in range(n)]
    
    df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )

    # 每个任务的微调后准确率（对角线）
    df["MeanFinetuneAccuracy"] = np.diag(matrix)

    # 每个任务最终准确率（最后一行）
    df["MeanFinalAccuracy"] = matrix[-1]
    print(len(matrix[-1]))

    mean_avg_acc = []
    for i in range(n):
        mean_avg_acc.append(np.mean(matrix[i, :i+1]))
    df["MeanAverageAccuracy"] = mean_avg_acc

    bwt = []
    for i in range(n):
        bwt.append(matrix[-1, i] - matrix[i, i])
    df["BackwardTransfer"] = bwt


    # 增加一行作为整体的性能指标
    overall = {
        col: np.nan for col in df.columns
    }
    overall["MeanFinetuneAccuracy"] = np.mean(np.diag(matrix))
    overall["MeanFinalAccuracy"] = np.mean(matrix[-1])
    overall["MeanAverageAccuracy"] = np.mean(mean_avg_acc)
    overall["BackwardTransfer"] = np.mean(bwt)
    df.loc["overall"] = overall

    print(df)
    save_path = osp.join(args.eval_dir, "metric_matrix.csv")
    df.to_csv(save_path)


if __name__ == "__main__":
    main()

    