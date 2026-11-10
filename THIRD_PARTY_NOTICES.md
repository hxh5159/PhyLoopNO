# Sources and retained licenses

This standalone source tree was extracted from [hxh5159/CDLNO-w](https://github.com/hxh5159/CDLNO-w) at commit `0eece431e4a2019256b048b5f7f67da1283f4b64`. Its original MIT license is retained in [LICENSE](LICENSE). The source repository was not modified or replaced, and its development history is not imported into this repository.

The task organization, original Physics-Attention/Transolver models and benchmark pipelines derive from [thuml/Transolver](https://github.com/thuml/Transolver), read at `75e0f67643806a81cd1d3f6adc88dd8c02416fe7`, accompanying *Transolver: A Fast Transformer Solver for PDEs on General Geometries*, ICML 2024, Haixu Wu, Huakun Luo, Haowen Wang, Jianmin Wang and Mingsheng Long.

The reproduced LinearNO implementation follows [HiPRL/LinearNO](https://github.com/HiPRL/LinearNO), read at `3f2b80df13c17a09e250f2ebe4d4ecdfd4acf269`, and *Transolver is a Linear Transformer: Revisiting Physics-Attention through the Lens of Linear Attention*, Wenjie Hu, Sidun Liu, Peng Qiao, Zhenglun Sun and Yong Dou, AAAI 2026 / arXiv:2511.06294v3. Task-specific projection, temperature, rank, initialization and training differences are retained and described in [REPRODUCING.md](docs/REPRODUCING.md).

AirfRANS pipeline components derive from [Extrality/AirfRANS](https://github.com/Extrality/AirfRANS) through the Transolver source tree. Their separately supplied license is retained verbatim at [Airfoil-Design-AirfRANS/LICENSE](Airfoil-Design-AirfRANS/LICENSE). Dataset licenses are separate from model source licenses. Download Standard PDE, AirfRANS and ShapeNet-Car data from their original providers; no dataset or downloaded checkpoint is redistributed here.

## Extraction scope

Retained: final V5 PhyLoopNO computation; pure LinearNO and original Transolver baselines; six Standard PDE and two industrial data/train/evaluation pipelines; strict checkpoint and epoch continuation infrastructure; field, Q/K and dense-expert figure generation; relevant synthetic tests.

Excluded: CDLNO/CDPA, KCDNO, MSAR-LNO, history/AttnRes families, loop V1-V4, latent-FFN and bilateral-adapter experiments, unused 3D model wrapper, old model launchers/configurations, planning prompts, conversation exports, historical audit evidence, archives, personal run directories, generated figures and monitoring helpers.

Only the Transolver and LinearNO baselines implemented in the source repository are shipped. Other methods whose published numbers appear in manuscript tables are references, not implementations fabricated for this release.
