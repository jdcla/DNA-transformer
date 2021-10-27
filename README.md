**This is a heavily adapted fork of the Transformer-XL repository ([https://github.com/kimiyoung/transformer-xl](https://github.com/kimiyoung/transformer-xl))**


This repository includes the code applied for the following two articles

1. ---
>**Novel transformer networks for improved sequence labeling in genomics**
>
>Jim Clauwaert, Willem Waegeman
>
>https://ieeexplore.ieee.org/document/9246292

2. ---
>**Explainability in transformer models for functional genomics**
>
>Jim Clauwaert, Gerben Menschaert, Willem Waegeman
>
>https://academic.oup.com/bib/article/22/5/bbab060/6214646?login=true
    ---

## Novel transformer networks for improved sequence labeling in genomics

**Data**

    data
    ├── eco_regulon
    │   ├── eco_regulon_70.npy
    ├── eco_TIS
    │   ├── eco_TIS.npy
    ├── eco_SMRTmeth
    │   ├── eco_m4c.npy


**Train models**

    run_m4c.sh
    run_TIS.sh
    run_TSS.sh
---
## Explainability in transformer models for functional genomics

**Data**

    data
    ├── eco_regulon
    │   ├── eco_regulon_strong.npy
    ├── eco_capseq
    │   ├── eco_rm_capseq.npy
    ├── eco_smrtcapseq
    │   ├── eco_smrt_capseq.npy
    ├── eco_SEndseq
    │   ├── eco_SEndseq.npy
    ├── eco_custom
    │   ├── eco_custom.npy

**Train models**

    run_regulon.sh
    run_cappable.sh
    run_smrtcappable.sh
    run_send.sh
    run_custom.sh
