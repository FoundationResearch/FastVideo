"""Rewrite a one-line user idea into 6 segment prompts using a CANDIDATE
dreamverse rewriter system prompt, on Groq gpt-oss-120b (direct OpenAI-compatible
call; enhancement is OFF on the render side, so this is the only rewrite).

Usage: python oe_rewrite.py <candidate_system_prompt.md> "<user_idea>" <out_dir>
Writes <out_dir>/segment_prompts.json = {idea, segment_prompts:[6], raw}.
Requires GROQ_API_KEY in env.
"""
import json
import os
import re
import sys

from openai import OpenAI

CAND_MD, IDEA, OUT_DIR = sys.argv[1], sys.argv[2], sys.argv[3]
BASE_URL = os.environ.get("FASTVIDEO_PROMPT_GROQ_API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL = os.environ.get("FASTVIDEO_PROMPT_GROQ_MODEL", "openai/gpt-oss-120b")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    system_prompt = open(CAND_MD).read()
    # Strip HTML comments (the EVOLVE-BLOCK markers + the evolve-region note) so
    # those meta-instructions never leak into the task model's system prompt.
    system_prompt = re.sub(r"<!--.*?-->", "", system_prompt, flags=re.DOTALL).strip()

    client = OpenAI(base_url=BASE_URL, api_key=os.environ["GROQ_API_KEY"])
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": IDEA}],
        temperature=0.7,
        response_format={"type": "json_object"},
        reasoning_effort="low",
    )
    content = resp.choices[0].message.content
    data = json.loads(content)
    segs = data.get("segment_prompts") or data.get("prompts") or data.get("seed_prompts")
    if not isinstance(segs, list) or len(segs) != 6:
        print(f"[rewrite] WARN expected 6 segment_prompts, got {len(segs) if isinstance(segs, list) else segs!r}",
              file=sys.stderr)
    out = os.path.join(OUT_DIR, "segment_prompts.json")
    json.dump({"idea": IDEA, "segment_prompts": segs, "raw": data}, open(out, "w"), indent=2)
    print(f"[rewrite] {len(segs) if isinstance(segs, list) else 0} segment prompts -> {out}")
    for i, p in enumerate(segs or []):
        print(f"  [{i}] {str(p)[:120]}")
    if not isinstance(segs, list) or len(segs) != 6:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
