from pathlib import Path
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from collections import Counter, defaultdict
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


class Retriever():
    def __init__(self, children, parents, configs):
        # building FAISS vector bases
        self.configs = configs
        self.children = children
        self.parents  = parents
        embeddings = HuggingFaceEmbeddings(
            model_name      = configs["DENSE_MODEL"],
            model_kwargs    = {"local_files_only": True} 
        )
    
        INDEX_DIR = Path(configs["INDEX_PATH"])      #Path("faiss_index") 
        if INDEX_DIR.exists():
            vectordb = FAISS.load_local(str(INDEX_DIR), embeddings)
        else:
            vectordb = FAISS.from_documents(children, embeddings)
            vectordb.save_local(str(INDEX_DIR))

        self.dense_retriever = vectordb.as_retriever(search_kwargs={"k": configs["DENSE_PICK"]})

        #---------------------------------------------------------
        corpus_tokens = [c.page_content.split() for c in children]
        self.bm25 = BM25Okapi(corpus_tokens)


    def bm25_retrieve_parents(self, query: str) -> Tuple[List[Document], List[Document]]:
        """
        return (child_hits, parent_hits)
            child_hits:  Document of the top-bm25_k child chunks with BM25 scores
            parent_hits: Document of the top_parent parent chunks (after deduplication)
        """
        # child score
        scores = self.bm25.get_scores(query.split())
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.configs["BM25_PICK"]]
        child_hits = [self.children[i] for i in top_idx]

        # mapped to parents
        parent_ids = []
        seen = set()
        for ch in child_hits:
            pid = ch.metadata["parent_id"]
            if pid not in seen:
                parent_ids.append(pid)
                seen.add(pid)
            if len(parent_ids) >= self.configs["TOP_PARENT"]:
                break
        parent_hits = [self.parents[i] for i in parent_ids]
        return child_hits, parent_hits


    def bm25_retrieve_text_parents(self, query: str):

        scores = self.bm25.get_scores(query.split())
        idx_sorted = sorted(range(len(scores)),
                            key=lambda i: scores[i], reverse=True)

        # filterring
        child_hits, parent_ids = [], []
        for i in idx_sorted:
            if len(child_hits) >= self.configs["k_child"]:
                break
            pid = self.children[i].metadata["parent_id"]
            if self.parents[pid].metadata["type"] != "parent":
                continue                           # skip media child chunks
            child_hits.append(self.children[i])
            parent_ids.append(pid)
            if len(set(parent_ids)) >= self.configs["k_parent"]:   # early stop
                break

        parent_hits = [self.parents[i] for i in set(parent_ids)]
        return child_hits, parent_hits


    def merge_parents(
        self,
        dense_parents: List[Document],
        sparse_parents: List[Document]
    ) -> List[Document]:
        """
        把稠密 + 稀疏父块合并，按 (book_idx,page_idx,type) 去重，
        保留顺序：先 dense，再 sparse 中新增的。
        """
        merged = []
        seen   = set()                                      # 去重键集合
        for p in dense_parents + sparse_parents:
            key = (p.metadata["book_idx"],
                   p.metadata["page_idx"],
                   p.metadata["type"])
            if key not in seen:
                merged.append(p)
                seen.add(key)
        return merged
        

    def hybrid_retrieval(self, query: str):
        # dense retrival
        dense_child_hits = self.dense_retriever.get_relevant_documents(query)
        dense_parent_ids = {c.metadata["parent_id"] for c in dense_child_hits}
        dense_parents    = [self.parents[i] for i in dense_parent_ids]
        dense_counter    = Counter(d.metadata["type"] for d in dense_parents)

        print(
            f"稠密检索到 {len(dense_child_hits)} 个子块，映射到 {len(dense_parents)} 个父块，"
            f"包含 {dense_counter.get('parent',0)+dense_counter.get('text',0)} 段文本，"
            f"{dense_counter.get('image',0)} 张图像，"
            f"{dense_counter.get('table',0)} 个表格，"
            # f"{dense_counter.get('equation',0)} 个公式"
        )

        # sparse retrival
        sparse_child_hits, sparse_parents = self.bm25_retrieve_text_parents(query)
        sparse_counter = Counter(d.metadata["type"] for d in sparse_parents)

        print(
            f"稀疏检索到 {len(sparse_child_hits)} 个子块，映射到 {len(sparse_parents)} 个父块，"
            f"包含 {sparse_counter.get('parent',0)+sparse_counter.get('text',0)} 段文本，"
            f"{sparse_counter.get('image',0)} 张图像，"
            f"{sparse_counter.get('table',0)} 个表格，"
            # f"{sparse_counter.get('equation',0)} 个公式"
        )

        results = self.merge_parents(dense_parents, sparse_parents)
        
        return results
        

    def related_equs(self, top_text_parents, docs):
        pages_of_text = {
            (p.metadata["book_idx"], p.metadata["page_idx"])
            for p in top_text_parents
        }

        eq_by_loc = defaultdict(list)          # key = (book,page) → list[Document]
        for d in docs:                         # docs = load_corpus() 得来的四类原始条目
            if d.metadata["type"] == "equation":
                key = (d.metadata["book_idx"], d.metadata["page_idx"])
                eq_by_loc[key].append(d)

        # 把这些页里的所有 equation 拉进来
        top_equations = []
        seen_eq_ids   = set()            # 防止同页同式重复
        for loc in pages_of_text:
            for eq in eq_by_loc.get(loc, []):
                # 你如果给 equation 生成过唯一 id，可在这里去重
                if id(eq) not in seen_eq_ids:
                    top_equations.append(eq)
                    seen_eq_ids.add(id(eq))

        print(f"\n关联到同页公式 {len(top_equations)} 条")
        self.preview_equations(top_equations)

        return top_equations


    def preview_equations(self, eq_list, n=3):
        for i, d in enumerate(eq_list[:n], 1):
            latex = (d.page_content[:100] + "…") if len(d.page_content) > 100 else d.page_content
            print(f"{i}. (equation, p{d.metadata['page_idx']})  {latex}")
















