import os
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', default='logs_S2A_NVAS/subset_1.0',type=str)
    parser.add_argument('--output-dir', type=str,
                        help='e.g. eval_99')
    args = parser.parse_args()
    return args


def read_metrics_json(path):
    try:
        with open(path, "r") as f:
            m = json.load(f)
        out = {
            "mag": float(m.get("mag", 0.0)),
            "lre": float(m.get("lre", 0.0)),
            "env": float(m.get("env", 0.0)),
            "infer_time": float(m.get("infer_time", 0.0)),
            "num_samples": int(m.get("num_samples", 0)),
        }
        return out
    except Exception:
        return None


if __name__ == "__main__":
    args = parse_args()

    args.output_dir= "eval_99"

    env_dict = {
        "office": list(range(1, 6)),
        "house": list(range(6, 9)),
        "apartment": list(range(9, 12)),
        "outdoor": list(range(12, 14)),
    }

    print("===================================")
    avg_dict = {}

    for env_name, env_ids in env_dict.items():
        metrics_list = []

        print(f"\n[{env_name}]")
        for i in env_ids:
            metrics_path = os.path.join(args.log_dir, f"{i}", args.output_dir, "metrics.json")
            m = read_metrics_json(metrics_path)
            if m is None:
                print(f"  room {i:02d} : MISSING ({metrics_path})")
                continue
            print(
                f"  room {i:02d} | "
                f"mag {m['mag']:.6f} | "
                f"lre {m['lre']:.6f} | "
                f"env {m['env']:.6f} | "
                f"infer_time {m['infer_time']:.6f} | "
                f"num {m['num_samples']}"
            )
            metrics_list.append(m)

        if len(metrics_list) == 0:
            print(f"  -> no valid rooms in {env_name}")
            continue

        keys = metrics_list[0].keys()
        env_avg = {k: sum(x[k] for x in metrics_list) / len(metrics_list) for k in keys}

        print(f"  -> {env_name} average", end=" ")
        for k in keys:
            print(f"{k} {env_avg[k]:.6f}", end=" ")
            avg_dict.setdefault(k, []).append(env_avg[k])
        print()

    print("\n-----------------------------------")
    print("average", end=" ")
    for k in avg_dict.keys():
        overall = sum(avg_dict[k]) / len(avg_dict[k])
        print(f"{k} {overall:.6f}", end=" ")
    print()
    print("===================================")
