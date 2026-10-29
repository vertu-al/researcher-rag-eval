# Scope: Researcher RAG Eval

## Project statement

`researcher-rag-eval` is a local, reproducible evaluation toolkit for
evidence-grounded RAG answers.

It evaluates whether a RAG answer makes claims that are traceable to supplied
evidence, uses valid citations, and abstains when the available evidence does
not support a substantive answer.

The project does not build a chatbot or a full retrieval system. It evaluates
the quality of already-produced RAG responses.

## Research question

> Given an answer, its claim-level citations, and a set of retrieved evidence
> passages, can a deterministic evaluation contract reliably identify
> unsupported claims, invalid citations, insufficient citation coverage, and
> failures to abstain?

## Motivation

A RAG system can fail in at least two different ways:

1. It may retrieve insufficient or irrelevant evidence.
2. It may receive appropriate evidence but still produce an unsupported answer.

An answer that contains citation markers is not necessarily grounded. A
citation may not exist, may not support the associated claim, or may leave
important claims without support. This project makes those distinctions
explicit and testable.

## V0 objective

V0 implements a small, deterministic citation-and-evidence contract over
hand-curated evaluation cases.

Each case will contain:

- a research question;
- evidence passages with stable identifiers;
- a candidate answer;
- atomic claims extracted from that answer;
- citations attached to individual claims;
- human-authored reference annotations describing whether each claim is
  supported, unsupported, or should have been abstained from.

The evaluator will compare the candidate response against this contract and
produce an auditable report.

## Initial evaluation dimensions

| Dimension | Question |
| --- | --- |
| Citation validity | Does every cited evidence identifier exist in the supplied evidence set? |
| Citation coverage | Does each substantive claim include at least one citation? |
| Evidence support | Is a cited passage annotated as supporting the associated claim? |
| Unsupported claims | Does the answer make a substantive claim without adequate supporting evidence? |
| Appropriate abstention | When the evidence is insufficient, does the answer explicitly decline to make an unsupported conclusion? |

## Initial failure taxonomy

The first benchmark cases will deliberately include:

- a fully supported answer;
- a claim with a nonexistent citation identifier;
- a claim cited to an irrelevant passage;
- a partially supported answer that overstates its evidence;
- an unsupported answer that should have abstained;
- an unnecessary abstention despite sufficient evidence.

These are controlled cases, not naturally occurring model outputs. Their purpose
is to test and document the evaluation logic before applying it to real RAG
runs.

## What V0 will implement

- a versioned schema for evidence-grounded RAG evaluation cases;
- deterministic checks for citation validity and citation coverage;
- support and abstention checks against explicit human reference annotations;
- machine-readable evaluation reports;
- unit tests for each failure mode;
- a small, fully local example benchmark.

## Out of scope for V0

- document ingestion, chunking, embeddings, or vector databases;
- retrieval-model training or benchmarking;
- LLM inference, API calls, or an LLM-as-judge;
- cloud deployment, dashboards, user accounts, or a chatbot interface;
- automatic determination of scientific truth;
- claims of clinical, legal, or scientific decision safety.

## Relationship to related work

This repository is separate from a RAG application or cloud evaluation registry.

A RAG application produces answers and retrieved evidence. An evaluation
registry stores and compares evaluation runs. `researcher-rag-eval` defines and
executes the local, transparent evaluation contract that makes those runs
meaningful.

## Reproducibility principles

- Evaluation cases and reference annotations are version-controlled.
- Checks are deterministic and require no external model or API.
- Every result records the evaluator version and case identifiers.
- Tests cover expected successes and deliberate failures.
- Limitations are documented alongside reported results.

## V0 success criteria

V0 is complete when a user can run one local command that:

1. evaluates a small benchmark of controlled cases;
2. produces a readable summary and machine-readable report;
3. correctly distinguishes all deliberately constructed failure modes; and
4. passes the unit-test suite.

## Non-claims

Passing this benchmark will not demonstrate that a RAG system is generally
truthful, that citations are semantically correct without human reference
annotations, or that the toolkit guarantees safe research assistance.

It demonstrates a transparent and reproducible way to make a defined subset of
groundedness and abstention failures observable.