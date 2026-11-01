## 源适配层，parser，处理输入的文件


import re
import json
import hashlib
from pathlib import Path
from docx import Document


# ============ 加载 ============

def load_docx_text(path):
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def split_pages(text):
    """返回 [(page_num, line), ...]，去掉分页符本身"""
    lines = text.splitlines()
    result = []
    current_page = 1
    for line in lines:
        m = re.match(r'=====\s*Page\s+(\d+)\s*=====', line.strip())
        if m:
            current_page = int(m.group(1))
            continue
        result.append((current_page, line))
    return result


# ============ 按题号切块 ============

def split_pages_and_questions(text):
    """
    不依赖题号（Word 自动编号读不出来），改用两个锚点切块：
    1. 块之间有空行分隔；
    2. 每个块必须包含 "(N分)" 或 "（N分）" 且包含选项或答案。
    题号用序号 1,2,3... 替代，不用原始题号。
    """
    page_pattern = re.compile(r'=====\s*Page\s+(\d+)\s*=====')
    page_markers = [(m.start(), int(m.group(1))) for m in page_pattern.finditer(text)]

    def get_page(pos):
        page = 1
        for m_pos, p in page_markers:
            if m_pos < pos:
                page = p
            else:
                break
        return page

    # 按空行切块，同时记录每块的起始位置（用于推断页码）
    blocks_raw = []
    cursor = 0
    for chunk in re.split(r'\n\s*\n', text):
        idx = text.find(chunk, cursor)
        if idx < 0:
            idx = cursor
        cursor = idx + len(chunk)
        chunk_clean = page_pattern.sub('', chunk).strip()
        if chunk_clean:
            blocks_raw.append((idx, chunk_clean))

    blocks = []
    q_counter = 0
    for start_pos, chunk in blocks_raw:
        # 必须包含分值标记
        if not re.search(r'[(（]\s*\d+\s*分\s*[)）]', chunk):
            continue
        # 必须包含选项或答案
        if not re.search(r'[A-D][.．、]|[选擇]\s*[A-D]', chunk):
            continue
        q_counter += 1
        blocks.append((q_counter, get_page(start_pos), chunk))

    return blocks

# ============ 字段提取 ============

# 答案模式：优先匹配 (N分)X / (N分)正确X，再匹配 选X
ANSWER_PATTERNS = [
    # (2分)A / （2分）A / (2分)正确A / (2分A（缺右括号）/ (2分  )A
    re.compile(r'[(（]\s*\d+\s*分\s*[)）]?\s*(?:正确)?\s*([A-D]+)'),
    # 选A / 选择A
    re.compile(r'[选擇]\s*([A-D]+)'),
]

# 选项前缀噪声：OC. -> C.，O B. -> B.，0 D. -> D.
NOISE_PREFIX_PATTERNS = [
    (re.compile(r'\bO\s*([A-D])[.．、]'), r'\1.'),
    (re.compile(r'\bO([A-D])[.．、]'), r'\1.'),
    (re.compile(r'\b0\s*([A-D])[.．、]'), r'\1.'),
]

OPTION_PATTERN = re.compile(r'([A-D])[.．、]\s*')


def extract_answer_and_body(content):
    """返回 (answer, 去掉答案和答题记录后的正文)"""
    for pat in ANSWER_PATTERNS:
        m = pat.search(content)
        if m:
            answer = m.group(1).upper()
            before = content[:m.start()]
            after = content[m.end():]
            # 剥掉答案后紧跟的答题记录字母，如 " D A. VPN" 里的孤立 D
            after = re.sub(r'^\s*[A-D]\s+(?=[A-D][.．、])', '', after)
            return answer, before + after
    return "", content


def clean_noise(text):
    for pat, repl in NOISE_PREFIX_PATTERNS:
        text = pat.sub(repl, text)
    return text


def extract_options_and_stem(text):
    """返回 (options_dict, stem)"""
    text = clean_noise(text)
    matches = list(OPTION_PATTERN.finditer(text))
    if not matches:
        return {}, text.strip()

    stem = text[:matches[0].start()].strip()
    # 剥离题干末尾孤立的答题记录字母（如第1题的 D）
    stem = re.sub(r'\s+[A-D]\s*$', '', stem).strip()

    options = {}
    for i, m in enumerate(matches):
        letter = m.group(1).upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        opt_text = text[start:end].strip()
        opt_text = re.sub(r'\s+', ' ', opt_text).strip()
        opt_text = re.sub(r'[）)]\s*$', '', opt_text).strip()  # 清理末尾噪声括号
        if opt_text and letter not in options:
            options[letter] = opt_text

    return options, stem


def determine_type(answer, options):
    if answer in ("正确", "错误"):
        return "judge"
    if answer and len(answer) >= 2 and all(c in "ABCD" for c in answer):
        return "multiple"
    return "single"


def detect_issues(stem, answer, options, qtype):
    issues = []
    if not stem or len(stem) < 5:
        issues.append("empty_stem")
    if not answer:
        issues.append("answer_missing")
    if qtype != "judge" and not options:
        issues.append("options_missing")
    if qtype != "judge" and options:
        letters = sorted(options.keys())
        if letters and letters[0] != "A":
            issues.append("options_not_starting_from_A")
        expected = [chr(ord('A') + i) for i in range(len(letters))]
        if letters != expected:
            issues.append("option_letter_gap")
        # 疑似 OCR：选项文本以自己的字母开头（如 D 选项文本是 "DIDA"）
        for letter, opt in options.items():
            if opt and len(opt) > 1 and opt[0] == letter:
                issues.append("suspected_ocr")
                break
    return issues


# ============ 组装 questions ============

def parse_question(qnum, page_num, content):
    answer, body = extract_answer_and_body(content)
    options, stem = extract_options_and_stem(body)
    qtype = determine_type(answer, options)
    issues = detect_issues(stem, answer, options, qtype)

    question_id = f"p{page_num}_{qnum:03d}"

    return {
        "question_id": question_id,
        "query": stem,
        "question_type": qtype,
        "answer": answer,
        "options": options,
        "gold_doc_ids": [f"{question_id}_k"],
        "source_locator": f"page:{page_num},q:{qnum}",
        "issues": issues,
    }


# ============ 派生 statements ============

def make_statement(q, source_file):
    qid = q["question_id"]
    answer = q["answer"]
    options = q["options"]
    stem = q["query"]

    answer_texts = []
    if answer and options:
        for letter in answer:
            if letter in options:
                answer_texts.append(f"{letter}.{options[letter]}")
    elif answer in ("正确", "错误"):
        answer_texts.append(answer)

    if answer_texts:
        answer_str = "；".join(answer_texts)
        content = f"针对问题「{stem}」，正确答案是：{answer_str}。"
    else:
        content = f"针对问题「{stem}」，正确答案是：{answer}。"

    content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
    token_count = max(1, len(content) // 2)

    return {
        "record_id": f"{qid}_k",
        "content": content,
        "content_type": "qa",
        "token_count": token_count,
        "content_hash": content_hash,
        "source_file": source_file,
        "source_locator": q["source_locator"],
        "quality_flags": q["issues"],
    }


# ============ 主流程 ============

def main(input_path, questions_out, statements_out):
    text = load_docx_text(input_path)
    blocks = split_pages_and_questions(text)   # ← 只改这一行
    print(f"Total blocks: {len(blocks)}")
    print(f"First 5 qnums: {[b[0] for b in blocks[:5]]}")
    print(f"Last 5 qnums: {[b[0] for b in blocks[-5:]]}")
    print(f"Total blocks: {len(blocks)}")
    print(f"All qnums: {[b[0] for b in blocks]}")
    if len(blocks) < 10:
        print("=== DEBUG text[:2000] ===")
        print(repr(text[:2000]))

    questions = []
    statements = []
    source_file = Path(input_path).name

    for qnum, page_num, content in blocks:
        if len(content.strip()) < 3:
            continue
        q = parse_question(qnum, page_num, content)
        if not q["answer"]:
            print(f"[NO ANSWER] q{qnum} page{page_num}: {repr(content[:150])}")
        questions.append(q)
        if q["answer"]:
            statements.append(make_statement(q, source_file))

    Path(questions_out).write_text(
        json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    Path(statements_out).write_text(
        json.dumps(statements, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Total blocks: {len(blocks)}")
    print(f"Total questions parsed: {len(questions)}")
    print(f"Total statements generated: {len(statements)}")

    issues_count = {}
    for q in questions:
        for iss in q["issues"]:
            issues_count[iss] = issues_count.get(iss, 0) + 1
    print(f"Issues summary: {issues_count}")

    clean_count = sum(1 for q in questions if not q["issues"])
    print(f"Clean questions (no issues): {clean_count}")


if __name__ == "__main__":
    main(
        input_path="./data/documents/30题的测试集.docx",
        questions_out="./data/output/questions.json",
        statements_out="./data/output/statements.json",
    )