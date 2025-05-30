import textwrap
from typing import List
from .utils import block_fmt 
from langchain.docstore.document import Document


CAPTION_PROMPT = """
You will be given a scientific image. 

1. **First**, decide which one of the following five categories it belongs to:
   - statistical_plot  
   - visualization_figure  
   - pipeline_diagram  
   - table
   - other_image  

2. **Then**, generate a detailed, precise description according to the category you chose, following these templates:

- **If statistical_plot**  
  1. Overview: Describe the purpose of the plot, including what is being measured and compared.  
  2. Axes Explanation: Describe the meaning of each axis, including respective logarithmic scales.  
  3. Legend Interpretation: Explain what each curve/line represents and how each one performs relative to each other.  
  4. Trends Description: Highlight significant distributions or trends, focusing on rates of change, inflection points, or other significant features.

- **If visualization_figure**  
  1. Overview: Introduce the purpose of the visualization, explaining what kind of data or experimental results are being presented.  
  2. Key Features: Describe the key visual components (axes, colors, markers, or spatial distributions) and explain their significance.  
  3. Methodological Insights: If applicable, explain how different methods, models, or experimental conditions are being compared.  
  4. Implications: Discuss the conclusions that can be drawn directly from the visualization; do not fabricate conclusions without supporting evidence.

- **If pipeline_diagram**  
  1. Overall Description: Summarize the goal of the pipeline, its major modules, and how they work together.  
  2. Step-by-Step Explanation: Describe the sequence of operations.  
  3. Key Modules and Functions: Highlight important elements such as backbone, data inputs/outputs, and main algorithms.  
  4. Diagram Details: Explain how colors, arrows, and blocks represent operations and data flow.

- **If table**  
  1. Table Overview: Describe the purpose of the table, including key information and research context.  
  2. Content Description: Explain what the table represents (experimental results, statistical summaries, parameter comparisons, etc.).  
  3. Structure Explanation: Describe what each row and column represent.  
  4. Key Findings: Highlight notable trends, relationships, or interpretations.
  
- **If other_image**  
  1. Overview: Describe the purpose of the image (the concepts, objects, or phenomena it represents).  
  2. Scientific Context: Explain which aspect of the study it illustrates.  
  3. Observations: Highlight notable details, relationships, or interpretations that can be drawn directly; do not fabricate conclusions.

**Output** only the final caption after you’ve chosen the right category.
"""


REWRITE_PROMPT = """
你将得到：
1. 原始 **AnswerText**（来自 DeepSeek Reasoner）。
2. 最多 5 个 **Media**（Image 或 Table）的描述，每个带唯一占位名 <MEDIA_i>。
3. 每个 Media 的 **Caption** （图/表标题）。

目标：  
- 对 AnswerText 进行润色，使行文连贯、信息丰富，但不得改变事实。  
- 判断哪些段落需要插入哪些 Media；请在相应段落 **行尾** 插入 `<MEDIA_i>` 占位符，可多张图表插同段，也可某些图表不使用。  
- 输出 **JSON**，字段：  
  ```json
  {
    "enhanced_paragraphs": ["第一段", "第二段", ...],
    "unused_media": ["<MEDIA_3>", ...]      // 若全部用完给 []
  }
注意：
不要改动 <MEDIA_i> 占位符格式。
不要生成除 JSON 之外的任何额外内容。
请务必输出 纯 JSON，不要加 ```json 包裹，也不要出现未转义的反斜杠，例如 \( \] 等。
"""


def build_retrieval_prompt(
    query: str,
    top_text_parents: List[Document],
    top_media_parents: List[Document],
    top_equations: List[Document]
) -> str:
    """
    根据检索出的文本、图表和公式父块，生成最终发给 LLM 的 Prompt 字符串。
    """
    # A) 文本父块
    text_section = "\n\n".join(
        block_fmt(d, i + 1) 
        for i, d in enumerate(top_text_parents)
    )
    # B) 图表父块
    media_section = "\n\n".join(
        block_fmt(d, i + 1) 
        for i, d in enumerate(top_media_parents)
    )
    # C) 公式父块
    eq_section = "\n\n".join(
        block_fmt(d, i + 1) 
        for i, d in enumerate(top_equations)
    )

    # 组装最终 prompt
    prompt = textwrap.dedent(f"""
        ## Query
        {query}

        ## Retrieved Context - TEXT (Top {len(top_text_parents)})
        {text_section}

        ## Retrieved Context - IMAGE/TABLE (Top {len(top_media_parents)})
        {media_section}

        ## Retrieved Context - EQUATION (same pages)
        {eq_section}

        ## Instruction
        你是专业的技术写作助手，请阅读 **Retrieved Context**，并仅基于这些信息回答 **Query**。
        - 如果某个信息未在检索结果出现，请说明“在提供的资料中未找到相关信息”。
        - 回答用简体中文，结构清晰，必要时分条列出。
    """).strip()

    return prompt