# Researcher RAG Eval

A small, reproducible toolkit for evaluating whether evidence-grounded RAG
answers make supported claims, use valid citations, and abstain when available
evidence is insufficient.

> **Research question:** Given an answer, its claim-level citations, and a set
> of retrieved evidence passages, can a transparent deterministic contract
> identify unsupported claims, invalid citations, missing citation coverage, and
> failures to abstain?

## Why this exists

A RAG response can look reliable while still failing in important ways:

- it can cite an evidence identifier that does not exist;
- it can attach a real but irrelevant citation to a claim;
- it can make a substantive claim without citing evidence;
- it can overstate what the retrieved evidence supports; or
- it can answer confidently when the available evidence should trigger
  abstention.

`researcher-rag-eval` makes these failure modes explicit, versioned, and
testable.

## V0: controlled citation and abstention benchmark

V0 evaluates hand-curated, synthetic cases. Each case contains:

- a question;
- retrieved evidence passages with stable identifiers;
- a candidate answer;
- atomic substantive claims and their citations;
- human reference annotations specifying whether each claim is supported; and
- an expected abstention decision.

The evaluator checks five dimensions:

| Dimension | Question |
| --- | --- |
| Citation validity | Does every cited evidence identifier exist? |
| Citation coverage | Does a substantive claim include a citation? |
| Evidence support | Does the citation point to evidence annotated as supporting the claim? |
| Unsupported claims | Does the answer make a claim marked unsupported by the reference annotation? |
| Appropriate abstention | Does the response abstain exactly when the controlled case requires it? |

## Example run

```bash
researcher-rag-eval --cases benchmark/controlled_cases.json