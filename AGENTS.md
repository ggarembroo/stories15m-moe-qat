# Agent protocol

Owner rule, revised 2026-11-23.

## When to use a second agent

Use a second agent for technical work:

- code changes
- distillation and training
- quantization and QAT
- experiment design
- debugging and architecture decisions

The second agent is a review layer. Send it the facts, metrics, hypotheses and
open questions. Ask for the most effective next path, the main flaw and the
decision rule. Accept its recommendation unless it contradicts a measured fact.
If it contradicts a measurement, say which fact conflicts.

## When not to use it

Do not spawn a second agent for:

- file searches
- lyrics or text edits
- routine retrieval
- simple formatting
- tasks the owner did not ask to review

## Cancellation

Stop or ignore any agent whose task is no longer relevant. Do not keep a search
agent alive after the file or text has been found.

## Counter

Report the review counter only for technical review cycles. Keep it in
`docs/AGENT_COUNTER.txt`.
