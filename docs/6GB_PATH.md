# The 6 GB path for MiniCPM-V 4.7 35B-A3B

## The arithmetic

35.2B total, 32.2B routed experts, 91.5 percent of the model. At 6 GB total with
10 percent headroom, the weight file cap is 5.116 GB, about 1.163 bpw overall.

| format | raw GB | working GB | fits |
|---|---:|---:|---|
| binary experts + 4-bit non-experts | 5.84 | 6.79 | no |
| binary experts + 2-bit non-experts | 5.09 | 5.96 | yes, 40 MB spare |
| ternary experts + 2-bit non-experts | 9.22 | 10.51 | no |
| 2T+4B mixed | 6.84 | 7.52 | no |
| 1T+3B per-expert | 6.49 | 7.14 | no |

Ternary experts alone are 6.36 GB at 1.58 bpw and 8.38 GB at 2.08 bpw. Ternary
35B all-resident in 6 GB is impossible.

The only all-resident 35B point is binary experts + 2-bit non-experts. It is also
the worst quality point: binary 25.17 PPL and routing collapse on the 36M
testbed, and 2-bit vision/embeddings damage OCR and multilingual quality.

## Three real paths

### 1. Ternary 9.1 GB + capped expert paging

The file is 9.1 GB, but the phone limit is PSS, not file size. The 12.9 GB Q2_K
already runs with PSS about 2.4 GB. A capped cache can keep PSS under 6 GiB.

Resident budget at 6 GiB:

| component | 4k | 8k |
|---|---:|---:|
| app, ART, GPU, surfaces | 450 MB | 450 MB |
| runtime, graph, scratch | 300 MB | 300 MB |
| dense weights, pinned | 1500 MB | 1500 MB |
| expert hot cache | 2900 MB | 2800 MB |
| ternary KV | 110 MB | 220 MB |
| vision encoder, transient | 300 MB | 300 MB |
| tokenizer, logits, index, slack | 100 MB | 100 MB |
| total | 5660 MB | 5670 MB |

Expected decode: 3 to 4 tok/s at about 45 percent expert hit rate. Naive mmap
fails because the kernel caches the whole 9.1 GB file and PSS climbs to 8 to
9 GB. Use O_DIRECT or mmap plus `madvise(MADV_DONTNEED)` on cold ranges, and
`MADV_RANDOM` to kill readahead.

Prefill is the dangerous transient. Stream each layer's expert region through a
16 to 32 MB buffer, do not fault the whole mmap. Load the vision encoder on
demand and cap the image at 448 to 896 px.

KV: keep K at q8_0 or q4_0, V ternary. Ternary K collapses long-context
retrieval.

### 2. Prune 28 to 32 percent, then ternary QAT

24 to 26B total, 5.2 to 5.6 GB, all-resident. Expected 85 to 92 percent of FP.
Requires REAP/HOPE pruning, router KD, and 5 to 20B QAT tokens. This is the only
all-resident path to a genuinely smart 6 GB model. Risk: the pruning knee.

### 3. Distill a 7 to 10B student at 4 to 6 bpw

4.6 to 5.5 GB, expected 90 to 95 percent of the teacher. Best absolute quality
at 6 GB, but a different model. Cost estimate from the agent: 12 to 15k
H100-hours, 26 to 35k USD, 2 to 4 weeks on 32 H100.

## Recommendation

Test path 1 first. It needs no training, uses the existing 35B weights, and keeps
ternary quality. The only open question is real PSS and tok/s on the phone.

Test protocol:

1. Push the 21 GB Q4 file or build the 9.1 GB ternary file.
2. Measure `smaps_rollup` Pss, Pss_Anon, Pss_File during load, prefill, vision,
   and decode.
3. Log `read_bytes` and major faults per token, expert hit rate per layer.
4. Check `logcat -b events` for `am_kill` and the MIUI kill reason.
5. Pass if peak PSS stays below 6 GiB, decode is at least 2 tok/s, and 4k
   retrieval does not collapse.

If paging fails, path 2 is the all-resident fallback. If quality is the priority
and budget exists, path 3.
