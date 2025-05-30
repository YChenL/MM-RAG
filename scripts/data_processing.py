import os, json
from tqdm import tqdm
from typing import List
from .prompts import CAPTION_PROMPT
from dashscope import MultiModalConversation
from langchain.docstore.document import Document
from concurrent.futures import ThreadPoolExecutor, as_completed



# 初始化captioner
qwen_key = os.getenv("DASHSCOPE_API_KEY")
if not qwen_key:
    raise RuntimeError("请先确保 DASHSCOPE_API_KEY 已正确设置并激活了 mmrag 环境")

def img_cap(image_path):
    messages = [{"role": "system",
                "content": [{"text": "You are a helpful assistant for image captioning. Think step by step."}]},
                {'role':'user',
                'content': [{'image': image_path},   
                            {'text': CAPTION_PROMPT}
                           ]}]
    response = MultiModalConversation.call(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
        api_key=qwen_key,
        model='qwen2.5-vl-7b-instruct', #'qwen2-vl-2b-instruct' free
        messages=messages,
        vl_high_resolution_images=False)

    return response["output"]["choices"][0]["message"].content[0]["text"]




def _process_image_inst(inst, IMAGE_ROOT):
    """
    辅助函数：给一个 inst 调用 img_cap 并返回一个 Document
    """
    img_path = os.path.join(IMAGE_ROOT, inst["img_path"])
    cap_list = inst.get("img_caption") or []
    descrip_list = [img_cap(img_path).strip()]
    return Document(
        page_content=" ".join(cap_list + descrip_list),
        metadata={
            "type":        "image",
            "book_idx":    inst.get("book_idx", -1),
            "page_idx":    inst.get("page_idx", -1),
            "img_path":    img_path,
            "img_caption": cap_list,
            "img_descrip": descrip_list,
            **{k: inst.get(k) for k in ("img_footnote",) if inst.get(k) is not None}
        }
    )


def _process_table_inst(inst, IMAGE_ROOT):
    """
    辅助函数：给一个 inst 调用 img_cap 并返回一个 Document
    """
    img_path = os.path.join(IMAGE_ROOT, inst["img_path"])
    cap_list = inst.get("table_caption") or []
    descrip_list = [img_cap(img_path).strip()]
    return Document(
        page_content=" ".join(cap_list + descrip_list),
        metadata={
            "type":        "table",
            "book_idx":    inst.get("book_idx", -1),
            "page_idx":    inst.get("page_idx", -1),
            "img_path":    img_path,
            "table_caption": cap_list,
            "table_descrip": descrip_list,
            **{b: inst.get(b) for b in ("table_body",) if inst.get(b) is not None},
            **{k: inst.get(k) for k in ("table_footnote",) if inst.get(k) is not None}
        }
    )




def load_corpus(KB_PATH, IMAGE_ROOT, parallel_image_workers: int = 16) -> List[Document]:
    docs: List[Document] = []
    with open(KB_PATH, encoding="utf-8") as f:
        all_insts = json.load(f)

    # 1) 先把非 image 类型的都处理好
    image_insts = []
    for inst in all_insts:
        t = inst.get("type")
        if t == "text":
            docs.append(Document(
                page_content=inst["text"],
                metadata={
                    "type":     "text",
                    "book_idx": inst.get("book_idx", -1),
                    "page_idx": inst.get("page_idx", -1),
                    **{k: inst.get(k) for k in ("text_level",) if inst.get(k) is not None}
                }
            ))
        elif t == "equation":
            docs.append(Document(
                page_content=inst["text"],
                metadata={
                    "type":     "equation",
                    "book_idx": inst.get("book_idx", -1),
                    "page_idx": inst.get("page_idx", -1),
                    **{k: inst.get(k) for k in ("text_format",) if inst.get(k) is not None}
                }
            ))
        elif t == "table":
            try:
                img_path = os.path.join(IMAGE_ROOT, inst.get("img_path",""))
                cap_list = inst.get("table_caption") or []
                body_list = [inst.get("table_body")] or []
                docs.append(Document(
                    page_content=" ".join(cap_list + body_list),
                    metadata={
                        "type":          "table",
                        "book_idx":      inst.get("book_idx", -1),
                        "page_idx":      inst.get("page_idx", -1),
                        "img_path":      img_path,
                        "table_caption": cap_list,
                        "table_body":    body_list,
                        **{k: inst.get(k) for k in ("table_footnote",) if inst.get(k) is not None}
                    }
                ))
            except:
                print(inst)
                
        elif t == "image":
            image_insts.append(inst)
        # 其它类型若有，可继续 elif

    # 2) 并行处理所有 image_inst
    with ThreadPoolExecutor(max_workers=parallel_image_workers) as ex:
        # 用 as_completed 可以加进度条
        futures = { ex.submit(_process_image_inst, inst, IMAGE_ROOT): inst for inst in image_insts }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Captioning images"):
            try:
                docs.append(fut.result())
            except Exception as e:
                inst = futures[fut]
                print(f"[Error] image inst {inst.get('img_path')} caption failed: {e}")

    print(f"Loaded {len(docs)} documents")
    return docs



def load_corpus_parallel(KB_PATH, IMAGE_ROOT, parallel_image_workers: int = 16) -> List[Document]:
    """
    读取 KB_PATH → 解析四类 inst → 并行调用 Qwen-VL
      · image  : 调用 _process_image_inst → caption+description
      · table  : 调用 _process_table_inst → caption+LLM description
      · text   : 直接写入
      · equation: 直接写入
    返回统一的 docs 列表
    """
    docs: List[Document] = []
    # ------- 读取原始 JSON -------
    with open(KB_PATH, encoding="utf-8") as f:
        all_insts = json.load(f)

    # ------- 按类型分桶 -------
    image_insts, table_insts = [], []
    for inst in all_insts:
        t = inst.get("type")
        if t == "text":
            docs.append(
                Document(
                    page_content=inst["text"],
                    metadata={
                        "type":     "text",
                        "book_idx": inst.get("book_idx", -1),
                        "page_idx": inst.get("page_idx", -1),
                        **{k: inst.get(k) for k in ("text_level",) if inst.get(k) is not None}
                    },
                )
            )
        elif t == "equation":
            docs.append(
                Document(
                    page_content=inst["text"],
                    metadata={
                        "type":     "equation",
                        "book_idx": inst.get("book_idx", -1),
                        "page_idx": inst.get("page_idx", -1),
                        **{k: inst.get(k) for k in ("text_format",) if inst.get(k) is not None}
                    },
                )
            )
        elif t == "image":
            image_insts.append(inst)
        elif t == "table":
            table_insts.append(inst)
        # 其它类型可继续 elif

    # ------- 并行处理 image & table -------
    media_total = len(image_insts) + len(table_insts)
    if media_total:
        with ThreadPoolExecutor(max_workers=parallel_image_workers) as ex:
            fut2inst = {}

            # 提交任务
            for inst in image_insts:
                fut2inst[ex.submit(_process_image_inst, inst, IMAGE_ROOT)] = ("image", inst)
            for inst in table_insts:
                fut2inst[ex.submit(_process_table_inst, inst, IMAGE_ROOT)] = ("table", inst)

            # 收集结果
            for fut in tqdm(
                as_completed(fut2inst),
                total=media_total,
                desc="Captioning media (image+table)",
            ):
                typ, inst = fut2inst[fut]
                try:
                    docs.append(fut.result())
                except Exception as e:
                    print(f"[Error] {typ} inst {inst.get('img_path')} caption failed: {e}")

    print(f"Loaded {len(docs)} documents")
    return docs
