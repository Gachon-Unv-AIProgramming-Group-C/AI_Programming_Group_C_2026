import json
import time
import csv
import os
import sys
import argparse
from datetime import datetime
from urllib import request, error as urllib_error

MCP_URL = os.environ.get("MCP_URL", "http://localhost:8000/mcp")

def call_mcp(question, response, options):
    payload = {
        "jsonrpc": "2.0",
        "id": int(time.time() * 1000),
        "method": "tools/call",
        "params": {
            "name": "check_hallucination",
            "arguments": {"question": question, "response": response, **options},
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(MCP_URL, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=120) as res:
        body = json.loads(res.read().decode("utf-8"))
    raw = body.get("result", {}).get("content", [{}])[0].get("text", "")
    if not raw:
        raise ValueError(f"Empty response: {body}")
    return json.loads(raw)

def load_dataset(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows

def save_results(results, out_dir, timestamp):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"results_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    csv_path = os.path.join(out_dir, f"results_{timestamp}.csv")
    if results:
        fields = [k for k in results[0] if k != "raw_result"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in results:
                writer.writerow({k: r[k] for k in fields})
    return json_path, csv_path

def evaluate(results):
    total = len(results)
    if total == 0:
        return {}
    tp = fp = tn = fn = uncertain = errors = 0
    correct = 0
    for r in results:
        if r.get("error"):
            errors += 1
            continue
        verdict = r.get("verdict", "")
        label = r["label"]
        if verdict == "UNCERTAIN":
            uncertain += 1
            continue
        pred_h = verdict == "HALLUCINATION"
        actual_h = label == "hallucination"
        if pred_h == actual_h:
            correct += 1
        if actual_h and pred_h: tp += 1
        elif not actual_h and pred_h: fp += 1
        elif not actual_h and not pred_h: tn += 1
        else: fn += 1
    decided = tp + fp + tn + fn
    accuracy  = correct / decided if decided > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {
        "total": total, "errors": errors, "uncertain": uncertain, "decided": decided, "correct": correct,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": round(accuracy, 4), "precision": round(precision, 4),
        "recall": round(recall, 4), "f1": round(f1, 4),
    }

def print_metrics(m):
    print("\n" + "=" * 50)
    print(f"  Total: {m['total']}  Errors: {m['errors']}  Uncertain: {m['uncertain']}  Decided: {m['decided']}")
    print(f"  Accuracy: {m['accuracy']:.4f}  Precision: {m['precision']:.4f}  Recall: {m['recall']:.4f}  F1: {m['f1']:.4f}")
    print(f"  TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}")
    print("=" * 50)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset.csv")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--t1", type=float, default=0.3)
    parser.add_argument("--t-star", type=float, default=0.7)
    parser.add_argument("--t2", type=float, default=0.5)
    parser.add_argument("--use-hf", action="store_true")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ids", nargs="+", type=int, default=None)
    args = parser.parse_args()

    dataset_path = args.dataset
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(os.path.dirname(__file__), dataset_path)

    rows = load_dataset(dataset_path)
    if args.ids:
        rows = [r for r in rows if int(r["id"]) in args.ids]
    if args.limit:
        rows = rows[:args.limit]

    options = {
        "targetModel": args.model, "verifierModel": args.model,
        "m": args.samples, "t1": args.t1, "tStar": args.t_star,
        "t2": args.t2, "useHuggingFaceNli": args.use_hf,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []

    print(f"Samples: {len(rows)}  model={args.model}  m={args.samples}  endpoint={MCP_URL}\n")

    for i, row in enumerate(rows, 1):
        sys.stdout.write(f"[{i:>3}/{len(rows)}] id={row['id']:<3} {row['label']:<15} {row.get('category',''):<15} ")
        sys.stdout.flush()
        start = time.time()
        try:
            res = call_mcp(row["question"], row["response"], options)
            elapsed = time.time() - start
            verdict = res.get("verdict", "UNKNOWN")
            pred_h = verdict == "HALLUCINATION"
            actual_h = row["label"] == "hallucination"
            mark = "✓" if verdict == "UNCERTAIN" else ("✓" if pred_h == actual_h else "✗")
            print(f"→ {verdict:<16} conf={res.get('confidence',0):.3f} stage={res.get('details',{}).get('stage','?')} {mark} ({elapsed:.1f}s)")
            results.append({
                "id": row["id"], "question": row["question"], "response": row["response"],
                "label": row["label"], "category": row.get("category",""),
                "verdict": verdict, "confidence": res.get("confidence",""),
                "stage": res.get("details",{}).get("stage",""),
                "layers_run": json.dumps(res.get("details",{}).get("layersRun",[])),
                "final_score": res.get("details",{}).get("final_score",""),
                "score1": res.get("details",{}).get("score1",""),
                "score2": res.get("details",{}).get("score2",""),
                "score3": res.get("details",{}).get("score3",""),
                "mode": res.get("details",{}).get("mode",""),
                "reason": res.get("reason",""),
                "elapsed_s": round(elapsed, 2), "error": "",
                "raw_result": res,
            })
        except Exception as e:
            elapsed = time.time() - start
            print(f"→ ERROR ({elapsed:.1f}s): {e}")
            results.append({
                "id": row["id"], "question": row["question"], "response": row["response"],
                "label": row["label"], "category": row.get("category",""),
                "verdict": "ERROR", "confidence": "", "stage": "", "layers_run": "",
                "final_score": "", "score1": "", "score2": "", "score3": "",
                "mode": "", "reason": "", "elapsed_s": round(elapsed,2), "error": str(e),
                "raw_result": {},
            })
        if i < len(rows):
            time.sleep(args.delay)

    metrics = evaluate(results)
    print_metrics(metrics)
    json_path, csv_path = save_results(results, args.out_dir, timestamp)
    print(f"\nJSON   : {json_path}\nCSV    : {csv_path}")
    mpath = os.path.join(args.out_dir, f"metrics_{timestamp}.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump({"options": vars(args), "metrics": metrics, "timestamp": timestamp}, f, ensure_ascii=False, indent=2)
    print(f"Metrics: {mpath}")

if __name__ == "__main__":
    main()
