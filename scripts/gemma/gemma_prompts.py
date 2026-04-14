"""Natural-language relational prompt construction for Gemma-2B validation.

Generates prompts of the form:

    Alice works at Acme. Bob works at Initech. [...] Alice's employer is

where the continuation target is a single-token tail entity. Care is taken to
pick entities that the Gemma tokenizer encodes as a single BPE token each — the
prompt curator must validate this against the live tokenizer before running.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class RelationalPrompt:
    text: str
    corrupted_text: str           # Swaps the head entity in the query (for patching).
    target_token: str             # Clean target tail (single token).
    corrupted_target_token: str   # Corrupted target tail (single token).
    query_head_entity: str
    relation: str


RELATIONS = [
    # (relation_name, fact_template, question_template)
    ("employer",   "{head} works at {tail}.",              "{head}'s employer is"),
    ("city",       "{head} lives in {tail}.",              "{head}'s city is"),
    ("pet",        "{head}'s pet is a {tail}.",            "{head} owns a"),
]


def build_prompt_set(
    head_entities: Sequence[str],
    tail_entities_per_relation: dict,
    n_prompts: int = 500,
    facts_per_prompt: int = 4,
    seed: int = 0,
) -> List[RelationalPrompt]:
    rng = random.Random(seed)
    prompts: List[RelationalPrompt] = []
    for _ in range(n_prompts):
        rel_name, fact_tpl, q_tpl = rng.choice(RELATIONS)
        tails = tail_entities_per_relation[rel_name]
        heads = rng.sample(list(head_entities), facts_per_prompt)
        used_tails = rng.sample(list(tails), facts_per_prompt)

        facts = [fact_tpl.format(head=h, tail=t) for h, t in zip(heads, used_tails)]
        qi = rng.randrange(facts_per_prompt)
        head_q, tail_q = heads[qi], used_tails[qi]
        text = " ".join(facts) + " " + q_tpl.format(head=head_q)

        # Corrupted: swap the query head with another in-context head whose
        # corresponding tail differs from tail_q.
        candidates = [(h, t) for h, t in zip(heads, used_tails) if h != head_q and t != tail_q]
        if not candidates:
            continue
        head_q_corr, tail_q_corr = rng.choice(candidates)
        corr_text = " ".join(facts) + " " + q_tpl.format(head=head_q_corr)

        prompts.append(RelationalPrompt(
            text=text,
            corrupted_text=corr_text,
            target_token=" " + tail_q,
            corrupted_target_token=" " + tail_q_corr,
            query_head_entity=head_q,
            relation=rel_name,
        ))
    return prompts


# A small curated list. Callers should validate single-token encoding with the
# live Gemma tokenizer before use; if a token splits, drop it from the pool.
DEFAULT_HEADS = [
    "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry",
    "Ivy", "Jack", "Kate", "Liam", "Mia", "Noah", "Olive", "Paul",
]
DEFAULT_TAILS = {
    "employer": ["Google", "Amazon", "Apple", "Netflix", "Microsoft", "Meta", "Oracle", "Intel"],
    "city": ["Paris", "London", "Tokyo", "Berlin", "Madrid", "Rome", "Boston", "Seattle"],
    "pet": ["dog", "cat", "hamster", "parrot", "rabbit", "turtle", "goldfish", "snake"],
}


if __name__ == "__main__":
    ps = build_prompt_set(DEFAULT_HEADS, DEFAULT_TAILS, n_prompts=5)
    for p in ps:
        print("CLEAN    :", p.text, "->", p.target_token)
        print("CORRUPTED:", p.corrupted_text, "->", p.corrupted_target_token)
        print()
