#!/usr/bin/env python3
"""Node 1: Generate QA pairs from wiki text and build ob provenance simultaneously.

Input:  pipeline_v2 wiki data.jsonl (raw wiki text + ob provenance)
Output: qa_chatml/data.jsonl (pure ChatML, 3-5 QA pairs per document, one pair per line)
        qa_chatml/.ob/       (author→section→document-index provenance chain)

Each wiki document yields 3-5 QA pairs. All pairs from the same document
share the same ob provenance (author → section → document-index).
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_project_root))
_rust_ob = _project_root.parent / "rust-originblame" / "python" / "src"
if _rust_ob.is_dir():
    sys.path.insert(0, str(_rust_ob))

import ob
from _ob_native import compute_hash, index_document, register_section
from openai import AsyncOpenAI

try:
    from ob_util.parsers.mediawiki import _strip_markup
except ImportError:
    import re
    def _strip_markup(text: str) -> str:
        text = re.sub(r'\{[^}]+\}', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        return text.strip()

DEFAULT_WIKI = "benchmarks/results/pipeline_v2/huggingface-zhwiki-10000-ob/jsonl/data.jsonl"
DEFAULT_OUTPUT = "benchmarks/results/pipeline_qa/qa_chatml"
DEFAULT_ENV = ".env"
DEFAULT_CONCURRENCY = 20
DEFAULT_MAX_CHARS = 0
DEFAULT_MAX_DOCS = 50000
MIN_ANSWER_LEN = 20

PROMPT = """基于以下维基百科内容，生成3到5个高质量的中文问答对。

要求：
- 问题应涵盖不同方面（人物背景、事件细节、因果关系、概念解释等）
- 答案应详细完整，用2到4句话解释，不要只回答一个词或日期
- 所有内容必须基于原文，不要编造
- 按以下格式输出，每个问答对之间用空行分隔：

Q: 问题
A: 详细回答

Q: 问题
A: 详细回答

如果原文内容不足以生成3个，可以少生成，但至少生成1个有意义的问答对。

标题：{title}
内容：{text}"""


def load_env(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


def parse_qa_pairs(raw: str) -> list[tuple[str, str]]:
    """Parse multiple Q/A pairs from API response. Returns list of (question, answer)."""
    pairs = []
    current_q = ""
    current_a = ""
    for line in raw.strip().split("\n"):
        line = line.strip()
        if line.startswith("Q:") or line.startswith("Q："):
            if current_q and current_a:
                pairs.append((current_q, current_a))
            current_q = line[2:].strip()
            current_a = ""
        elif line.startswith("A:") or line.startswith("A："):
            current_a = line[2:].strip()
        elif current_a and line:
            current_a += " " + line
        elif current_q and line:
            pass
    if current_q and current_a:
        pairs.append((current_q, current_a))

    return [(q, a) for q, a in pairs
            if "无法生成" not in q and len(a) >= MIN_ANSWER_LEN]


class OBTracker:
    def __init__(self, ob_dir: str):
        ob.init(ob_dir=ob_dir)
        self.ob_dir = ob_dir
        self.author_cache: dict[str, str] = {}
        self.section_cache: dict[str, str] = {}
        self.n_authors = 0
        self.n_sections = 0
        self.n_tracked = 0
        self.n_skipped = 0

    def _resolve_ids(self, meta: list[dict]) -> list[str]:
        ids = []
        for a in meta:
            name = a.get("name", "")
            email = a.get("email", "")
            if not name:
                continue
            cache_key = f"{name}<{email}>"
            if cache_key not in self.author_cache:
                try:
                    self.author_cache[cache_key] = ob.author_add(
                        name, email, ob_dir=self.ob_dir,
                    )
                    self.n_authors += 1
                except Exception:
                    self.author_cache[cache_key] = cache_key
            ids.append(self.author_cache[cache_key])
        return ids

    def track(self, chatml: dict, wiki: dict) -> None:
        authors_meta = []
        contributors_meta = []
        try:
            authors_meta = json.loads(wiki.get("authors_json", "[]"))
        except (json.JSONDecodeError, TypeError):
            pass
        try:
            contributors_meta = json.loads(wiki.get("contributors_json", "[]"))
        except (json.JSONDecodeError, TypeError):
            pass

        if not authors_meta:
            self.n_skipped += 1
            return

        license_str = wiki.get("license", "CC-BY-SA-4.0")
        year_str = str(wiki.get("year", "2024"))
        title = wiki.get("title", "unknown").replace("/", "_")
        heading = wiki.get("heading", "")
        section_path = f"qa/{title}#{heading}" if heading else f"qa/{title}"

        author_ids = self._resolve_ids(authors_meta)
        if not author_ids:
            self.n_skipped += 1
            return

        contributor_ids = self._resolve_ids(contributors_meta)

        if section_path in self.section_cache:
            section_hash = self.section_cache[section_path]
        else:
            try:
                section_hash = register_section(
                    section_path, author_ids, contributor_ids, license_str, year_str, self.ob_dir,
                )
                self.section_cache[section_path] = section_hash
                self.n_sections += 1
            except Exception:
                self.n_skipped += 1
                return

        try:
            line_hash = compute_hash(chatml)
            index_document(self.ob_dir, line_hash, "data.jsonl", [section_hash])
            self.n_tracked += 1
        except Exception:
            self.n_skipped += 1


async def run(args):
    with open(args.wiki, encoding="utf-8") as f:
        wiki_docs = [json.loads(line) for line in f if line.strip()]
    total = min(len(wiki_docs), args.max_docs) if args.max_docs else len(wiki_docs)
    wiki_docs = wiki_docs[:total]

    data_jsonl = os.path.join(args.output, "data.jsonl")
    # Resume by doc index (not line count) since each doc now produces variable QA pairs
    resume_file = os.path.join(args.output, ".progress.json")
    if os.path.exists(resume_file):
        with open(resume_file) as f:
            progress = json.load(f)
        docs_done = progress.get("docs_done", 0)
    else:
        docs_done = 0
    remaining = wiki_docs[docs_done:]

    print(f"Wiki:   {args.wiki} ({total} docs)")
    print(f"Output: {data_jsonl}")
    print(f"Docs done: {docs_done}, Remaining: {len(remaining)}")

    if not remaining:
        print("All done.")
        return

    os.makedirs(args.output, exist_ok=True)
    ob_tracker = OBTracker(str(Path(args.output).resolve()))

    client = AsyncOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="not-needed",
        max_retries=1,
        timeout=120,
    )
    sem = asyncio.Semaphore(args.concurrency)
    output = open(data_jsonl, "a", encoding="utf-8")

    import time
    t0 = time.time()
    generated = 0
    failed = 0
    total_pairs = 0

    batch_size = args.concurrency * 4

    for batch_start in range(0, len(remaining), batch_size):
        batch_docs = remaining[batch_start : batch_start + batch_size]
        batch_global_idx = range(docs_done + batch_start, docs_done + batch_start + len(batch_docs))

        tasks = [_gen_one(client, wiki_docs[i], i, args.max_chars, sem) for i in batch_global_idx]
        results = await asyncio.gather(*tasks)

        for global_idx, result in zip(batch_global_idx, results):
            if result is None:
                failed += 1
                continue
            for q, a in result:
                chatml = {"messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ]}
                output.write(json.dumps(chatml, ensure_ascii=False) + "\n")
                ob_tracker.track(chatml, wiki_docs[global_idx])
                total_pairs += 1
            generated += 1

        output.flush()
        with open(resume_file, "w") as f:
            json.dump({"docs_done": docs_done + batch_start + len(batch_docs)}, f)

        elapsed = time.time() - t0
        rate = generated / elapsed if elapsed > 0 else 0
        remaining_count = len(remaining) - generated - failed
        eta = remaining_count / rate if rate > 0 else 0
        print(
            f"  {docs_done + generated}/{total} docs "
            f"({total_pairs} pairs) "
            f"rate={rate:.1f}docs/s fail={failed} ETA={eta / 60:.0f}min"
        )

    output.close()

    try:
        from _ob_native import clean as _ob_clean
        _ob_clean(ob_tracker.ob_dir, False)
    except Exception:
        pass

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed / 60:.0f}min. {generated} docs, {total_pairs} pairs, {failed} failed.")
    print(f"  Authors: {ob_tracker.n_authors}, Sections: {ob_tracker.n_sections}, "
          f"Tracked: {ob_tracker.n_tracked}, Skipped: {ob_tracker.n_skipped}")


async def _gen_one(client: AsyncOpenAI, doc: dict, idx: int, max_chars: int, sem: asyncio.Semaphore):
    text = doc.get("text", "")[:max_chars] if max_chars > 0 else doc.get("text", "")
    text = _strip_markup(text)
    title = doc.get("title", "未知")

    async with sem:
        for attempt in range(3):
            try:
                r = await client.chat.completions.create(
                    model="qwen/qwen3.5-9b",
                    messages=[{"role": "user", "content": PROMPT.format(title=title, text=text)}],
                    temperature=0.7,
                    max_tokens=600,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                pairs = parse_qa_pairs(r.choices[0].message.content)
                if pairs:
                    return pairs
            except Exception:
                if attempt == 2:
                    print(f"  [{idx}] FAILED")
                    return None
                await asyncio.sleep(2 ** attempt)
    print(f"  [{idx}] NO VALID PAIRS")
    return None


def main():
    parser = argparse.ArgumentParser(description="Node 1: Generate QA + build ob provenance")
    parser.add_argument("--wiki", default=DEFAULT_WIKI)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--env", default=DEFAULT_ENV)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--max-docs", type=int, default=DEFAULT_MAX_DOCS)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    args = parser.parse_args()

    load_env(args.env)
    print("Using local vLLM at http://localhost:8000/v1")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
