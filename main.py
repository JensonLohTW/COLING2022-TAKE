#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys


def _run(cmd, cwd=None):
    print("[*] " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def _add_common_take_args(parser):
    parser.add_argument("--dataset", type=str, default="tiage")
    parser.add_argument("--name", type=str, default="TAKE_tiage_all_feats")
    parser.add_argument("--use-centrality", dest="use_centrality", action="store_true")
    parser.add_argument("--no-centrality", dest="use_centrality", action="store_false")
    parser.set_defaults(use_centrality=True)
    parser.add_argument("--centrality-alpha", type=float, default=1.5)
    parser.add_argument("--centrality-feature-set", type=str, default="all", choices=["none", "imp_pct", "all"])
    parser.add_argument("--centrality-window", type=int, default=2)
    parser.add_argument("--node-id-json", type=str, default="datasets/tiage/node_id.json")
    parser.add_argument("--dgcn-predictions-dir", type=str, default=None)
    parser.add_argument("--edge-lists-dir", type=str, default=None)
    parser.add_argument("--node-mapping-csv", type=str, default=None)
    parser.add_argument("--base-data-path", type=str, default=None)
    parser.add_argument("--base-output-path", type=str, default=None)


def _build_take_cmd(args, mode):
    cmd = [sys.executable, "./TAKE/Run.py", "--name", args.name, "--dataset", args.dataset, "--mode", mode]
    if args.use_centrality:
        cmd.append("--use_centrality")
        cmd += ["--centrality_alpha", str(args.centrality_alpha)]
        cmd += ["--centrality_feature_set", args.centrality_feature_set]
        cmd += ["--centrality_window", str(args.centrality_window)]
        cmd += ["--node_id_json", args.node_id_json]
        if args.dgcn_predictions_dir:
            cmd += ["--dgcn_predictions_dir", args.dgcn_predictions_dir]
        if args.edge_lists_dir:
            cmd += ["--edge_lists_dir", args.edge_lists_dir]
        if args.node_mapping_csv:
            cmd += ["--node_mapping_csv", args.node_mapping_csv]
    if args.base_data_path:
        cmd += ["--base_data_path", args.base_data_path]
    if args.base_output_path:
        cmd += ["--base_output_path", args.base_output_path]
    return cmd


def cmd_export_centrality(args):
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "dgcn3_export_predictions.py"),
        "--dataset_name",
        args.dataset_name,
        "--alphas",
        args.alphas,
    ]
    if args.model_path:
        cmd += ["--model_path", args.model_path]
    if args.output_dir:
        cmd += ["--output_dir", args.output_dir]
    _run(cmd)


def cmd_train_take(args):
    cmd = _build_take_cmd(args, "train")
    _run(cmd, cwd=os.path.join(os.path.dirname(__file__), "knowSelect"))


def cmd_infer_take(args):
    cmd = _build_take_cmd(args, "inference")
    _run(cmd, cwd=os.path.join(os.path.dirname(__file__), "knowSelect"))


def cmd_ablation(args):
    base = dict(
        dataset=args.dataset,
        centrality_alpha=args.centrality_alpha,
        centrality_window=args.centrality_window,
        node_id_json=args.node_id_json,
        dgcn_predictions_dir=args.dgcn_predictions_dir,
        edge_lists_dir=args.edge_lists_dir,
        node_mapping_csv=args.node_mapping_csv,
        base_data_path=args.base_data_path,
        base_output_path=args.base_output_path,
    )

    configs = [
        ("TAKE_tiage_text_only", "none", False),
        ("TAKE_tiage_imp_pct", "imp_pct", True),
        ("TAKE_tiage_all_feats", "all", True),
    ]

    for name, feature_set, use_centrality in configs:
        args.name = name
        args.centrality_feature_set = feature_set
        args.use_centrality = use_centrality
        cmd_train_take(args)
        cmd_infer_take(args)


def cmd_pipeline(args):
    cmd_export_centrality(args)
    cmd_train_take(args)
    cmd_infer_take(args)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export-centrality")
    export_parser.add_argument("--dataset-name", type=str, default="tiage")
    export_parser.add_argument("--alphas", type=str, default="1.5")
    export_parser.add_argument("--model-path", type=str, default="")
    export_parser.add_argument("--output-dir", type=str, default="")
    export_parser.set_defaults(func=cmd_export_centrality)

    train_parser = subparsers.add_parser("train-take")
    _add_common_take_args(train_parser)
    train_parser.set_defaults(func=cmd_train_take)

    infer_parser = subparsers.add_parser("infer-take")
    _add_common_take_args(infer_parser)
    infer_parser.set_defaults(func=cmd_infer_take)

    ablation_parser = subparsers.add_parser("ablation")
    _add_common_take_args(ablation_parser)
    ablation_parser.set_defaults(func=cmd_ablation)

    pipeline_parser = subparsers.add_parser("pipeline")
    _add_common_take_args(pipeline_parser)
    pipeline_parser.add_argument("--dataset-name", type=str, default="tiage")
    pipeline_parser.add_argument("--alphas", type=str, default="1.5")
    pipeline_parser.add_argument("--model-path", type=str, default="")
    pipeline_parser.add_argument("--output-dir", type=str, default="")
    pipeline_parser.set_defaults(func=cmd_pipeline)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
