# Egor Pobozhii

Local-first AI  
Ternary LLMs, phone clusters, offline assistants

I build small models that run on hardware people already own  
I measure instead of guessing

## Now

- Ternary QAT for MoE
- Phone clusters with ggml-rpc
- Offline Android assistant

## Projects

| Project | What |
|---|---|
| [stories15m-moe-qat](https://github.com/ggarembroo/stories15m-moe-qat) | Ternary QAT pipeline for a 36M Mixtral MoE |
| [phone-cluster](https://github.com/ggarembroo/phone-cluster) | One local LLM across two Android phones |
| [autosplit-sweep](https://github.com/ggarembroo/autosplit-sweep) | Speed-aware tensor split sweep for ggml-rpc |
| [peht-architecture-whitepaper](https://github.com/ggarembroo/peht-architecture-whitepaper) | Defensive publication with DOI and OpenTimestamps |
| [krukoai-releases](https://github.com/ggarembroo/krukoai-releases) | Offline Android AI assistant |

## Numbers

| Experiment | Result |
|---|---|
| 36M MoE FP32 | 16.20 PPL |
| 36M MoE ternary PTQ | 3404.56 PPL |
| 36M MoE ternary QAT + outlier + calibration + mixed precision | 21.78 PPL |
| MiniCPM 35B-A3B Q3 mixed on 16 GB phone | 6.3 tok/s decode |
| MiniCPM 35B-A3B Q2_K on 16 GB phone | 12.9 GB, ~1 tok/s |

## Stack

PyTorch, transformers, llama.cpp, ggml, React Native, Android, CUDA, OpenCL

## Contact

GitHub [@ggarembroo](https://github.com/ggarembroo)
