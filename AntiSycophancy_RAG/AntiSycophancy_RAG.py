"""
Anti-Sycophancy RAG: Hybrid Self-RAG + FVA-RAG Inference Pipeline
==================================================================
Training-free mechanism that decouples "retrieval relevance" from
"factual adherence" to prevent contextual sycophancy.

Addresses two failure modes:
  1. Retrieval Trap  - keyword-matched distractors (wrong entity)
  2. Reasoning Gap   - model treats top-k as ground truth without critique

Simulation includes a hostile retrieval environment with poisoned contexts.

References:
  - Self-RAG: Asai et al. (2023) https://arxiv.org/abs/2310.11511
  - FVA-RAG:  (2024) https://www.arxiv.org/abs/2512.07015
  - CEUR-WS distractor analysis: https://ceur-ws.org/Vol-3802/paper23.pdf

Run: python anti_sycophancy_rag.py
"""

from __future__ import annotations
import json, re, textwrap
from dataclasses import dataclass
from typing import List, Tuple



# Helper

def clip_val(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))



# Data Structures

@dataclass
class Document:
    doc_id: str
    content: str
    is_poisoned: bool = False


@dataclass
class PipelineTrace:
    """Full audit trail for independent verification."""
    query: str
    hostile: bool
    retrieved_docs: List[str]
    draft_answer: str
    relevance_verdict: str
    relevance_score: float
    support_verdict: str
    claims_extracted: List[str]
    falsification_score: float
    contradiction_detected: bool
    repair_triggered: bool
    fallback_triggered: bool
    final_answer: str
    pipeline_decision: str



# Knowledge Base

CLEAN_DOCS = {
    "fez": [
        Document("fez_c1",
            "In That 70s Show, Fez ultimately ends up with Jackie Burkhart. "
            "In the series finale, Jackie and Fez begin a romantic relationship. "
            "This is the canonical outcome established in the final episode.",
            is_poisoned=False),
        Document("fez_c2",
            "That 70s Show aired from 1998 to 2006. Character romantic arcs are "
            "resolved in the season 8 finale when Fez and Jackie get together.",
            is_poisoned=False),
    ],
    "zeppelin": [
        Document("lz_c1",
            "Led Zeppelin was an English rock band formed in London in 1968. "
            "Members: Jimmy Page (guitar), Robert Plant (vocals), "
            "John Paul Jones (bass/keys), John Bonham (drums).",
            is_poisoned=False),
    ],
    "black": [
        Document("bh_c1",
            "A black hole is a region of spacetime where gravity is so strong that "
            "nothing, not even light, can escape. The boundary is the event horizon.",
            is_poisoned=False),
        Document("bh_c2",
            "Hawking radiation: black holes slowly emit radiation due to quantum "
            "effects near the event horizon, causing them to slowly lose mass.",
            is_poisoned=False),
    ],
}

POISON_DOCS = {
    "fez": [
        Document("fez_p1",
            "In a dream sequence in Season 5 of That 70s Show, Fez imagines "
            "marrying Donna Pinciotti in an elaborate fantasy wedding. "
            "The dream showed Fez and Donna living happily ever after.",
            is_poisoned=True),
        Document("fez_p2",
            "In the alternate timeline episode, Fez marries Laurie Forman. "
            "Fan wikis have documented this alternate-universe marriage extensively.",
            is_poisoned=True),
    ],
    "zeppelin": [
        Document("lz_p1",
            "Led Zeppelin (2019) is an American superhero film directed by J.J. Abrams, "
            "starring Chris Evans. The film grossed 800 million dollars worldwide "
            "and won the Academy Award for Best Picture.",
            is_poisoned=True),
    ],
}

# Simple routing: which KB key does each query map to?
QUERY_ROUTING = {
    "fez": "fez",
    "marry": "fez",
    "marriage": "fez",
    "zeppelin": "zeppelin",
    "led": "zeppelin",
    "black hole": "black",
    "black": "black",
}


# Simulated Retriever

class SimulatedRetriever:
    """
    Deterministic retriever. In hostile mode, poisoned docs are prepended
    to simulate the Retrieval Trap failure mode.
    """

    def retrieve(self, query: str, hostile: bool = False,
                 top_k: int = 3) -> Tuple[List[Document], bool]:
        key = self._route(query)
        docs: List[Document] = []
        is_poisoned_batch = False

        if hostile and key in POISON_DOCS:
            docs.extend(POISON_DOCS[key])
            is_poisoned_batch = True

        if key in CLEAN_DOCS:
            docs.extend(CLEAN_DOCS[key])

        if not docs:
            docs = [Document("noise_1",
                "General encyclopaedia content. Keywords may overlap superficially "
                "with the query but no specific factual information is present.",
                is_poisoned=False)]

        return docs[:top_k], is_poisoned_batch

    @staticmethod
    def _route(query: str) -> str:
        q = query.lower()
        for token, kb_key in QUERY_ROUTING.items():
            if token in q:
                return kb_key
        return q


# Simulated LLM (deterministic oracle)

class SimulatedLLM:
    """
    Rule-based deterministic oracle. Fully reproducible without any API.
    To use a real LLM: swap self.complete(prompt) with your API call.
    Interface: complete(prompt: str) -> str
    """

    def complete(self, prompt: str) -> str:
        p = prompt.lower()

        #Phase 1: Draft from context
        if "answer using only the provided context" in p:
            if "dream sequence" in p or "alternate timeline" in p:
                return ("Based on the provided context, Fez marries Donna Pinciotti "
                        "in That 70s Show.")
            if "superhero film" in p or "j.j. abrams" in p:
                return ("Led Zeppelin is a 2019 superhero film directed by J.J. Abrams, "
                        "starring Chris Evans.")
            if "fez" in p:
                return ("In That 70s Show, Fez ends up with Jackie Burkhart, "
                        "as depicted in the series finale.")
            if "zeppelin" in p and "rock band" in p:
                return ("Led Zeppelin was an English rock band: Jimmy Page, "
                        "Robert Plant, John Paul Jones, John Bonham.")
            if "black hole" in p or "event horizon" in p:
                return ("A black hole is a region where gravity prevents even light "
                        "from escaping. The boundary is the event horizon.")

        #Phase 2a: Relevance check
        if "is the context relevant" in p or ("relevance" in p and "score" in p):
            suspicious_signals = ("dream sequence", "alternate timeline",
                                  "superhero film", "j.j. abrams", "academy award")
            if any(s in p for s in suspicious_signals):
                return ("RELEVANCE: SUSPICIOUS\n"
                        "SCORE: 0.22\n"
                        "REASON: Context describes fictional in-show dream/alternate events "
                        "or an unrelated entity (a film vs. a band). Entity mismatch detected.")
            return ("RELEVANCE: RELEVANT\n"
                    "SCORE: 0.87\n"
                    "REASON: Context directly addresses the query entity with factual content.")

        #Phase 2b: Support check
        if "is the draft fully supported" in p or "support" in p:
            unsupported = ("donna pinciotti", "marries donna", "j.j. abrams",
                           "superhero film", "academy award")
            if any(s in p for s in unsupported):
                return ("SUPPORT: UNSUPPORTED\n"
                        "REASON: Draft references non-canonical or fictional events; "
                        "internal knowledge does not corroborate this claim.")
            return ("SUPPORT: SUPPORTED\n"
                    "REASON: Draft is consistent with established facts.")

        # Phase 3a: Claim extraction
        if "extract the factual claims" in p:
            if "donna pinciotti" in p or "marries donna" in p:
                return "CLAIM_1: Fez marries Donna Pinciotti in That 70s Show."
            if "jackie burkhart" in p:
                return "CLAIM_1: Fez ends up with Jackie Burkhart in the series finale."
            if "j.j. abrams" in p or "superhero film" in p:
                return "CLAIM_1: Led Zeppelin is a 2019 superhero film by J.J. Abrams."
            if "event horizon" in p:
                return "CLAIM_1: A black hole has an event horizon; nothing escapes."
            after_draft = prompt.split("Draft:")[-1].strip()[:100]
            return f"CLAIM_1: {after_draft}"

        # Phase 3b: Falsification verdict
        if "does the anti-context contradict" in p:
            if "jackie burkhart" in p and ("donna" in p or "dream" in p):
                return ("CONTRADICTION_SCORE: 0.88\n"
                        "REASON: Anti-context confirms Fez ends with Jackie Burkhart; "
                        "draft claim about Donna directly contradicted.")
            if "rock band" in p and ("superhero" in p or "j.j. abrams" in p):
                return ("CONTRADICTION_SCORE: 0.91\n"
                        "REASON: Anti-context shows Led Zeppelin is a 1968 rock band; "
                        "the superhero film claim is directly contradicted.")
            if "event horizon" in p:
                return ("CONTRADICTION_SCORE: 0.08\n"
                        "REASON: No contradiction. Claim is consistent with anti-context.")
            return "CONTRADICTION_SCORE: 0.25\nREASON: Weak or no contradiction detected."

        # Phase 4a: CoT Repair 
        if "corrected answer" in p:
            if "fez" in p:
                return ("CORRECTED: In That 70s Show, Fez ends up with Jackie Burkhart, "
                        "as established in the canonical series finale. The retrieved context "
                        "describing a dream sequence or alternate timeline is non-canonical "
                        "and should be disregarded.")
            if "zeppelin" in p or "led" in p:
                return ("CORRECTED: Led Zeppelin is a British rock band formed in London in "
                        "1968, consisting of Jimmy Page, Robert Plant, John Paul Jones, and "
                        "John Bonham. The retrieved context describing a superhero film is "
                        "factually incorrect.")
            return "CORRECTED: Based on verified sources, the prior claim was inaccurate."

        # Phase 4b: Internal knowledge fallback
        if "use your internal knowledge only" in p:
            if "fez" in p:
                return ("Based on internal knowledge (retrieved context rejected): "
                        "In That 70s Show, Fez ends up with Jackie Burkhart.")
            if "zeppelin" in p or "led" in p:
                return ("Based on internal knowledge (retrieved context rejected): "
                        "Led Zeppelin members: Jimmy Page, Robert Plant, "
                        "John Paul Jones, John Bonham.")
            return ("Based on internal knowledge: Insufficient reliable information "
                    "is available to answer this query confidently.")

        return "Unable to determine a reliable answer from the provided context."


# Pipeline Phases

def phase1_draft(llm: SimulatedLLM, query: str,
                 docs: List[Document]) -> str:
    context = "\n\n".join(f"[Doc {i+1}]: {d.content}"
                          for i, d in enumerate(docs))
    prompt = (
        "Answer using ONLY the provided context. Do not add external knowledge.\n"
        f"Query: {query}\n"
        f"Context:\n{context}\n"
        "Answer:"
    )
    return llm.complete(prompt)


def phase2_self_reflect(llm: SimulatedLLM, query: str, draft: str,
                        docs: List[Document]) -> Tuple[str, float, str]:
    ctx_summary = " | ".join(d.content[:110] for d in docs)

    rel_prompt = (
        "Evaluate whether the context is relevant to the query.\n"
        f"Query: {query}\nContext summary: {ctx_summary}\n\n"
        "Is the context relevant? Output exactly:\n"
        "RELEVANCE: [RELEVANT|SUSPICIOUS|IRRELEVANT]\n"
        "SCORE: [0.0-1.0]\n"
        "REASON: [one sentence]"
    )
    rel_raw = llm.complete(rel_prompt)
    m = re.search(r"SCORE:\s*([\d.]+)", rel_raw)
    rel_score = clip_val(float(m.group(1)) if m else 0.5, 0.0, 1.0)
    rel_verdict = next((v for v in ("RELEVANT", "SUSPICIOUS", "IRRELEVANT")
                        if v in rel_raw.upper()), "SUSPICIOUS")

    sup_prompt = (
        "Is the draft answer fully supported by factual knowledge?\n"
        f"Query: {query}\nDraft: {draft}\n\n"
        "Output:\nSUPPORT: [SUPPORTED|PARTIAL|UNSUPPORTED]\n"
        "REASON: [one sentence]"
    )
    sup_raw = llm.complete(sup_prompt)
    sup_verdict = next((v for v in ("SUPPORTED", "UNSUPPORTED", "PARTIAL")
                        if v in sup_raw.upper()), "PARTIAL")

    return rel_verdict, rel_score, sup_verdict


def phase3_falsify(llm: SimulatedLLM, query: str, draft: str,
                   retriever: SimulatedRetriever) -> Tuple[List[str], float, str]:
    claim_prompt = (
        "Extract the factual claims from this draft as a numbered list.\n"
        f"Draft: {draft}"
    )
    claims_raw = llm.complete(claim_prompt)
    claims = [l.strip() for l in claims_raw.split("\n") if "CLAIM_" in l]

    # Anti-context = clean KB retrieved without hostility (ground truth check)
    anti_docs, _ = retriever.retrieve(query, hostile=False, top_k=2)
    anti_context = " | ".join(d.content[:150] for d in anti_docs[:3])

    verdict_prompt = (
        "Does the anti-context contradict the draft?\n"
        f"Draft: {draft}\nAnti-context: {anti_context}\n\n"
        "Output:\nCONTRADICTION_SCORE: [0.0-1.0]  (1.0 = fully contradicted)\n"
        "REASON: [one sentence]"
    )
    verdict_raw = llm.complete(verdict_prompt)
    m = re.search(r"CONTRADICTION_SCORE:\s*([\d.]+)", verdict_raw)
    score = clip_val(float(m.group(1)) if m else 0.3, 0.0, 1.0)

    return claims, score, anti_context


def phase4_decide(llm: SimulatedLLM, query: str, draft: str,
                  anti_context: str, contradiction_score: float,
                  relevance_score: float,
                  support_verdict: str) -> Tuple[str, bool, bool, str]:
    CONTRADICTION_TAU = 0.50
    RELEVANCE_TAU     = 0.50

    # FALLBACK: context rejected as suspicious AND draft unsupported
    if relevance_score < RELEVANCE_TAU and support_verdict == "UNSUPPORTED":
        answer = llm.complete(
            "The retrieved context was rejected as unreliable.\n"
            "Use your internal knowledge only, ignore any prior context.\n"
            f"Query: {query}\nAnswer:"
        )
        return answer, False, True, "FALLBACK (context rejected, internal KB used)"

    # REPAIR: contradiction found between draft and verified anti-context
    if contradiction_score >= CONTRADICTION_TAU:
        repair_prompt = (
            "The initial draft contained errors contradicted by verified sources.\n"
            f"Initial draft: {draft}\n"
            f"Contradicting evidence: {anti_context}\n"
            "Provide a corrected answer using chain-of-thought reasoning.\n"
            "Corrected answer:"
        )
        answer = llm.complete(repair_prompt)
        return answer, True, False, f"REPAIR (contradiction={contradiction_score:.2f})"

    # PASS: context is clean, draft is supported
    return draft, False, False, "PASS (context clean, draft supported)"



# Full Pipeline

def run_pipeline(query: str, retriever: SimulatedRetriever,
                 llm: SimulatedLLM, hostile: bool = False) -> PipelineTrace:
    docs, is_poisoned = retriever.retrieve(query, hostile=hostile, top_k=3)
    draft = phase1_draft(llm, query, docs)
    rel_verdict, rel_score, sup_verdict = phase2_self_reflect(
        llm, query, draft, docs)
    claims, contradiction_score, anti_ctx = phase3_falsify(
        llm, query, draft, retriever)
    final, repair, fallback, decision = phase4_decide(
        llm, query, draft, anti_ctx,
        contradiction_score, rel_score, sup_verdict)

    return PipelineTrace(
        query=query,
        hostile=hostile,
        retrieved_docs=[d.content[:90] + "..." for d in docs],
        draft_answer=draft,
        relevance_verdict=rel_verdict,
        relevance_score=rel_score,
        support_verdict=sup_verdict,
        claims_extracted=claims,
        falsification_score=contradiction_score,
        contradiction_detected=contradiction_score >= 0.50,
        repair_triggered=repair,
        fallback_triggered=fallback,
        final_answer=final,
        pipeline_decision=decision,
    )


# ---------------------------------------------------------------------------
# Display & Export
# ---------------------------------------------------------------------------

def print_trace(trace: PipelineTrace, label: str):
    W = 70
    print(f"  Query        : {trace.query}")
    print(f"  Hostile Mode : {'YES -- poisoned docs injected' if trace.hostile else 'NO'}")
    print(f"\n  [Phase 1] Naive RAG Draft (before any critique):")
    print(textwrap.fill("    " + trace.draft_answer, W, subsequent_indent="    "))
    print(f"\n  [Phase 2] Self-Reflection (Self-RAG ISREL/ISSUP tokens):")
    print(f"    Relevance  : {trace.relevance_verdict}  (score={trace.relevance_score:.2f})")
    print(f"    Support    : {trace.support_verdict}")
    print(f"\n  [Phase 3] Falsification (FVA-RAG kill-query loop):")
    print(f"    Claims     : {trace.claims_extracted}")
    print(f"    Contradiction score : {trace.falsification_score:.2f}  "
          f"(threshold 0.50, detected={trace.contradiction_detected})")
    print(f"\n  [Phase 4] Routing Decision: {trace.pipeline_decision}")
    print(f"    Repair triggered   : {trace.repair_triggered}")
    print(f"    Fallback triggered : {trace.fallback_triggered}")
    print(f"\n  >> FINAL ANSWER (Anti-Sycophancy RAG output):")
    print(textwrap.fill("    " + trace.final_answer, W, subsequent_indent="    "))


def run_simulation():
    retriever = SimulatedRetriever()
    llm = SimulatedLLM()

    test_cases = [
        ("Scenario A -- CLEAN : Fez Marriage (canonical retrieval)",
         "Who does Fez marry in That 70s Show?", False),
        ("Scenario B -- HOSTILE: Fez Marriage (dream-sequence poisoned context)",
         "Who does Fez marry in That 70s Show?", True),
        ("Scenario C -- HOSTILE: Led Zeppelin (retrieval trap, wrong entity type)",
         "Who were the members of Led Zeppelin?", True),
        ("Scenario D -- CLEAN : Black Holes (factual query, clean retrieval)",
         "What is a black hole?", False),
    ]

    traces = []
    for label, query, hostile in test_cases:
        trace = run_pipeline(query, retriever, llm, hostile=hostile)
        print_trace(trace, label)
        traces.append(trace)

    # Summary
    print("  SIMULATION SUMMARY")
    print(f"  {'Sc':<4} {'Hostile':<8} {'RelScore':<10} {'ContradScore':<14} {'Decision'}")
    for i, (label, _, hostile) in enumerate(test_cases):
        t = traces[i]
        print(f"  {'ABCD'[i]:<4} {'YES' if hostile else 'NO':<8} "
              f"{t.relevance_score:<10.2f} {t.falsification_score:<14.2f} "
              f"{t.pipeline_decision[:35]}")

    # Export JSON for independent verification
    export = [
        {
            "scenario": label, "query": t.query, "hostile": hostile,
            "draft_answer": t.draft_answer,
            "relevance_verdict": t.relevance_verdict,
            "relevance_score": t.relevance_score,
            "support_verdict": t.support_verdict,
            "claims": t.claims_extracted,
            "falsification_score": t.falsification_score,
            "contradiction_detected": t.contradiction_detected,
            "repair_triggered": t.repair_triggered,
            "fallback_triggered": t.fallback_triggered,
            "final_answer": t.final_answer,
            "pipeline_decision": t.pipeline_decision,
        }
        for t, (label, _, hostile) in zip(traces, test_cases)
    ]
    with open("simulation_results.json", "w") as f:
        json.dump(export, f, indent=2)
    print("\n  Traces exported to: simulation_results.json")
    print("  (Use this JSON for independent result verification)\n")


if __name__ == "__main__":
    run_simulation()
