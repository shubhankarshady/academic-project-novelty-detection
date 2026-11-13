import json

with open("results/model_optimization/semantic_feasibility_study.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("="*100)
print(f"{'Case ID':35s} | {'Prod Ridge':15s} | {'Cand TF-IDF':15s} | {'Semantic Only':15s} | {'Hybrid (a=0.5)':15s}")
print("="*100)

for c in data["controlled_case_evaluations"]:
    cid = c["case_id"]
    m = c["method_evaluations"]
    r_prod = f"#{m['prod_ridge_tfidf']['target_rank']} ({m['prod_ridge_tfidf']['target_sim']:.2f})"
    r_cand = f"#{m['cand_constrained_tfidf']['target_rank']} ({m['cand_constrained_tfidf']['target_sim']:.2f})"
    r_sem = f"#{m['semantic_only']['target_rank']} ({m['semantic_only']['target_sim']:.2f})"
    r_hyb = f"#{m['hybrid_alpha_050']['target_rank']} ({m['hybrid_alpha_050']['target_sim']:.2f})"
    print(f"{cid:35s} | {r_prod:15s} | {r_cand:15s} | {r_sem:15s} | {r_hyb:15s}")

print("\n" + "="*100)
print("RETRIEVAL SUCCESS SUMMARY")
print("="*100)
for k, v in data["retrieval_summary_stats"].items():
    print(f"  {k:22s} -> Paraphrase Top-1: {v['paraphrase_cases_top1']}, Top-3: {v['paraphrase_cases_top3']}, Top-5: {v['paraphrase_cases_top5']}, Distractor Filtered: {v['distractor_filtered_top1']}")
