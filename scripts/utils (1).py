"""
Merge every *.json file under /data/huali_data/  (each is a list of instances)
into one big JSON list and save to /data/huali_mm/huali_corpus.json
"""
import json, os
from tqdm import tqdm
from pathlib import Path
from itertools import islice
from collections import Counter, defaultdict
from langchain.docstore.document import Document

# CORPUS_PATH = Path("/data/huali_data")
# OUTPUT_DIR  = Path("/data/huali_mm")
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# OUTPUT_FILE = OUTPUT_DIR / "huali_corpus.json"



def merge_corpus(CORPUS_PATH, OUTPUT_FILE):
    all_instances = []

    json_files = sorted(CORPUS_PATH.rglob("*.json"))  
    for book_idx, json_file in enumerate(json_files):
        with open(json_file, "r", encoding="utf-8") as f:
            try:
                items = json.load(f)          
                if isinstance(items, list):
                    for inst in items:
                        inst["book_idx"] = book_idx
                    all_instances.extend(items)
                else:
                    print(f"[Warn] {json_file} 不是列表，已跳过")
            except json.JSONDecodeError as e:
                print(f"[Error] {json_file} 解析失败: {e}")

    print(f"Total merged instances: {len(all_instances)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_instances, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved to {OUTPUT_FILE}")
    


def analyze_kb_types(kb_path):
    with open(kb_path, encoding="utf-8") as f:
        all_insts = json.load(f)

    type_counter = Counter()
    missing_img_path_idxs = []
    samples = defaultdict(list)

    for idx, inst in enumerate(tqdm(all_insts, desc="Analyzing KB instances")):
        t = inst.get("type")
        type_counter[t] += 1

        if t == "image" and not inst.get("img_path"):
            missing_img_path_idxs.append(idx)

        if len(samples[t]) < 5:
            samples[t].append(inst)

    print(f"\n一共发现 {len(type_counter)} 种不同的 type：")
    for t, cnt in type_counter.items():
        print(f"  - {t!r}: {cnt} 条")
    if missing_img_path_idxs:
        print(f"\n注意：有 {len(missing_img_path_idxs)} 条 type='image' 的记录缺少 'img_path'，示例索引：{missing_img_path_idxs[:10]}{'...' if len(missing_img_path_idxs)>10 else ''}")
    else:
        print("\n所有 type='image' 的记录均包含 'img_path'。")

    print("\n每种 type 的前 5 个实例（完整内容）：")
    for t, inst_list in samples.items():
        print(f"\n--- Type = {t!r} （共 {type_counter[t]} 条） ---")
        for i, inst in enumerate(inst_list, start=1):
            print(f"{i}. {inst}")
 


def union_docs(docs):
    unique_docs, seen = [], set()
    for d in docs:
        key = (d.metadata["book_idx"], d.metadata["page_idx"], d.page_content.strip())
        if key not in seen:
            unique_docs.append(d)
            seen.add(key)
    docs = unique_docs
    print(f"After dedup: {len(docs)}")



def save_docs(docs, OUTPUT_DOCS= Path("/data/huali_mm/docs.json")):
    serializable = [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in docs
    ]
    with open(OUTPUT_DOCS, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(docs)} docs to {OUTPUT_DOCS}")



def load_serialized_docs(path: Path):
    with open(path, encoding="utf-8") as f:
        raw_items = json.load(f)         # list[dict]

    # 重新构造成 Document
    docs = [
        Document(page_content=item["page_content"],
                 metadata=item["metadata"])
        for item in raw_items
    ]
    return docs


def preview_docs_by_type(docs, n_preview=5):
    """按 metadata['type'] 分组打印前 n_preview 个 Document"""
    buckets = defaultdict(list)
    for d in docs:
        buckets[d.metadata["type"]].append(d)
    
    for t in ["text", "image", "table", "equation"]:
        lst = buckets.get(t, [])
        print(f"\n=== {t.upper()}  (共 {len(lst)} 条) ===")
        for i, d in enumerate(islice(lst, n_preview), 1):
            print(f"{i}.", d)