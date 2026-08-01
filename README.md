# FasterGS: Workload-Balanced Gaussian Splatting for Faster Training and Rendering

**🎉 Accepted to ACM Multimedia 2026 (ACM MM 2026)**

Official code for **FasterGS**, a plug-and-play CUDA backend that rebalances irregular 3D Gaussian Splatting workloads to accelerate both training and rendering, without changing the underlying Gaussian representation.

> **Abstract.** 3DGS achieves high-quality real-time novel-view synthesis, yet GPU efficiency is often limited by heterogeneous workloads and sequential alpha compositing. FasterGS reshapes skewed Gaussian footprints and density distributions via *footprint-aware scheduling* (aggregate small Gaussians, decompose large ones) and *bucketized parallel local compositing*, preserving correctness and differentiability while reducing long-tail latency.

Across seven mainstream 3DGS pipelines and three standard benchmarks, FasterGS delivers roughly **1.56×** training speedup and **2.11×** FPS improvement on average, with comparable reconstruction quality.

---

## Highlights

- **Imbalance-aware Gaussian scheduling** — pack underfilled small-footprint Gaussians into warp-efficient packets; split large-footprint Gaussians into tile-bounded sub-tasks.
- **Bucketized parallel alpha compositing** — partition per-tile depth-ordered lists into fixed-size buckets, render in parallel, then merge with an associative rule (forward & backward).
- **Drop-in acceleration** — integrate into existing trainable pipelines.


### Why FasterGS?

| Capability | Schedule / pruning methods<br>(e.g. FastGS, Mini-Splatting) | Culling / footprint methods<br>(e.g. Speedy-Splat, FlashGS) | **FasterGS (Ours)** |
|---|:---:|:---:|:---:|
| Accelerates **training** | ✓ | △ | **✓** |
| Accelerates **rendering** | △ | ✓ | **✓** |
| Kernel-level **workload rebalancing** | ✗ | △ | **✓** |
| **Plug-and-play** across pipelines | △ | △ | **✓** |
| Quality largely **unchanged** | ✓ | ✓ | **✓** |
| Orthogonal to existing accelerators | — | — | **✓** |

✓ supported △ partial / method-dependent ✗ not the focus

### Plug-and-Play Gains (Mip-NeRF 360)

Replacing only the CUDA backend (`Vanilla` → `+Ours`). Time in minutes; numbers in parentheses are speedup / FPS ratio vs. the same baseline.

| Backbone | Time ↓ | FPS ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|
| 3DGS | 27.3 → **16.4** (1.7×) | 119 → **332** (2.8×) | 0.815 → 0.816 | 0.216 → 0.215 |
| Mini-Splatting | 21.0 → **10.3** (2.0×) | 387 → **778** (2.0×) | 0.822 → 0.819 | 0.217 → 0.221 |
| Taming-3DGS | 5.7 → **4.0** (1.4×) | 264 → **663** (2.5×) | 0.795 → 0.796 | 0.260 → 0.259 |
| DashGaussian | 9.3 → **6.8** (1.4×) | 146 → **326** (2.2×) | 0.820 → 0.820 | 0.213 → 0.213 |
| Speedy-Splat | 15.1 → **9.6** (1.6×) | 851 → **1159** (1.4×) | 0.786 → 0.784 | 0.287 → 0.291 |
| FastGS | 3.3 → **2.7** (1.2×) | 921 → **967** (1.1×) | 0.798 → 0.798 | 0.261 → 0.260 |
| Mini-Splatting2 | 3.2 → **2.2** (1.4×) | 337 → **775** (2.3×) | 0.821 → 0.823 | 0.215 → 0.208 |

Averaged over all 21 baseline–dataset pairs (Mip-NeRF 360, Deep Blending, Tanks & Temples): **~1.56×** training speedup and **~2.11×** FPS, with SSIM / LPIPS essentially unchanged. Full tables are in the paper.

---

## Repository Layout

```
FasterGS/
├── gs_submodules/                      # CUDA rasterization backends
│   ├── diff-gaussian-rasterization_faster/   # ★ FasterGS (ours)
│   ├── diff-gaussian-rasterization/          # Vanilla 3DGS
│   ├── diff-gaussian-rasterization-dash/     # DashGaussian
│   ├── diff-gaussian-rasterization-taming/   # Taming-3DGS
│   ├── diff-gaussian-rasterization_speedy/   # Speedy-Splat
│   ├── diff-gaussian-rasterization_fastgs/   # FastGS
│   ├── diff-gaussian-rasterization_ms/       # Mini-Splatting
│   ├── diff-gaussian-rasterization_ms_2/     # Mini-Splatting2
│   ├── simple-knn/
│   ├── fused-ssim/
│   └── lanczos-resampling/
├── 3DGS/                               # Kerbl et al., TOG 2023
├── DashGaussian/                       # Chen et al., CVPR 2025
├── taming-3dgs/                        # Mallick et al., SIGGRAPH Asia 2024
├── speedy-splat/                       # Hanson et al., CVPR 2025
├── FastGS/                             # Ren et al., CVPR 2026
├── mini-splatting/                     # Fang & Wang, ECCV 2024
└── mini-splatting2/                    # Fang & Wang, TPAMI 2026
```

Each method folder provides:

| Script | Role |
|--------|------|
| `full_eval.py` | Full train / render / metrics on Mip-NeRF 360, Tanks & Temples, Deep Blending |
| `auto_run.py` | Convenience launcher (Vanilla vs `--fastergs`) |
| `train.py` / `train_faster.py` | Single-scene training |
| `render.py` / `render_faster.py` | Rendering |
| `metrics.py` | SSIM / LPIPS evaluation |

Pass `--fastergs` to use our accelerated CUDA backend.

---

## Requirements

- Linux with NVIDIA GPU
- CUDA toolkit compatible with your PyTorch build (paper experiments: RTX 3090, CUDA toolkits 12.6)
- Standard 3DGS dependencies (`plyfile`, `tqdm`, etc.; follow each baseline’s original environment when possible)

### GPU Compatibility

Some methods may fail to build or run on certain GPU architectures (e.g. **Speedy-Splat** and **FasterGS** are known to be incompatible with NVIDIA RTX 5090). This is typically due to CUDA / compute-capability constraints in the original kernels. Please check the corresponding [official repositories](#acknowledgements) for supported GPUs, CUDA versions, and any upstream fixes before reporting issues.

---

## Installation

### 1. Clone

```bash
git clone https://github.com/GoogolplexGoodenough/FasterGS
cd FasterGS
```

### 2. Build shared helpers (once)

```bash
cd gs_submodules/simple-knn
pip install -e .

cd ../fused-ssim        # if used by the target baseline
pip install -e .

cd ../lanczos-resampling  # DashGaussian only
pip install -e .
```

### 3. Build FasterGS CUDA backend (ours)

```bash
cd gs_submodules/diff-gaussian-rasterization_faster
pip install -e .
```

### 4. Build the vanilla / baseline backends you need

Install only the rasterizers corresponding to the methods you will run, for example:

```bash
# Vanilla 3DGS
cd gs_submodules/diff-gaussian-rasterization && pip install -e .

# DashGaussian / Taming / Speedy / FastGS / Mini / Mini2
cd gs_submodules/diff-gaussian-rasterization-dash && pip install -e .
cd gs_submodules/diff-gaussian-rasterization-taming && pip install -e .
cd gs_submodules/diff-gaussian-rasterization_speedy && pip install -e .
cd gs_submodules/diff-gaussian-rasterization_fastgs && pip install -e .
cd gs_submodules/diff-gaussian-rasterization_ms && pip install -e .
cd gs_submodules/diff-gaussian-rasterization_ms_2 && pip install -e .
```

> **Note.** Keep only one active rasterizer package per Python environment when package names collide, or use separate conda envs per baseline if preferred.

---

## Datasets

We follow the standard 3DGS evaluation protocol on:

| Dataset | Scenes | Links |
|---------|--------|-------|
| [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) | bicycle, flowers, garden, stump, treehill, room, counter, kitchen, bonsai | [Paper](https://arxiv.org/abs/2111.12077) · [Data](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip) |
| [Tanks & Temples](https://www.tanksandtemples.org/) | truck, train | Preprocessed COLMAP packs from [3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) |
| [Deep Blending](https://repo-sam.inria.fr/fungraph/deep-blending/) | drjohnson, playroom | Same 3DGS release |

COLMAP-preprocessed Tanks & Temples / Deep Blending scenes used by 3DGS are available from the [GRAPHDECO 3DGS page](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).

Edit dataset roots in each method’s `auto_run.py` (or pass `-m360` / `-tat` / `-db` to `full_eval.py`):

```python
m360 = r'/path/to/mipnerf360'
tat  = r'/path/to/tandt'
db   = r'/path/to/db'
```

---

## Quick Start

### Run Vanilla vs FasterGS on one pipeline

Example for **3DGS** (same pattern for other folders):

```bash
cd 3DGS

# Edit auto_run.py dataset paths, then:
python auto_run.py
```

Or call evaluation directly:

```bash
# Vanilla
python full_eval.py \
  -m360 /path/to/mipnerf360 \
  -tat  /path/to/tandt \
  -db   /path/to/db \
  --output_path eval/3DGS

# + FasterGS
python full_eval.py --fastergs \
  -m360 /path/to/mipnerf360 \
  -tat  /path/to/tandt \
  -db   /path/to/db \
  --output_path eval/3DGS_faster
```

Useful flags (shared across most `full_eval.py`):

| Flag | Meaning |
|------|---------|
| `--fastergs` | Use FasterGS CUDA backend |
| `--skip_training` | Only render + metrics |
| `--skip_rendering` | Skip rendering |
| `--skip_metrics` | Skip metric computation |
| `--output_path` | Output directory |

### Other baselines

```bash
cd DashGaussian   && python auto_run.py
cd taming-3dgs    && python auto_run.py
cd speedy-splat   && python auto_run.py
cd FastGS         && python auto_run.py
cd mini-splatting/ms     && python auto_run.py
cd mini-splatting2/msv2  && python auto_run.py
```

Each `auto_run.py` launches Vanilla and `--fastergs` variants with the baseline’s original training schedule.

---

## Method Overview

```
Gaussian-centric preprocessing          Image-centric rasterization
┌─────────────────────────────┐         ┌──────────────────────────────┐
│ Footprint-aware rebalancing │         │ Two-stage bucketized α-blend │
│  • Small: aggregate → warps │         │  • Stage I: parallel buckets │
│  • Large: decompose → tiles │         │  • Stage II: associative merge│
└─────────────────────────────┘         └──────────────────────────────┘
```

See the paper for kernel-level profiling, component ablations (LGD / SGA / BAC), and per-iteration forward/backward breakdowns.

---

## Acknowledgements

This repository builds upon and compares against the following excellent works. Please cite them if you use the corresponding code:

| Method | Venue | Code | Project |
|--------|-------|------|---------|
| [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) | TOG / SIGGRAPH 2023 | [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) | [Page](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) |
| [Mini-Splatting](https://arxiv.org/abs/2403.14166) | ECCV 2024 | [fatPeter/mini-splatting](https://github.com/fatPeter/mini-splatting) | — |
| [Taming 3DGS](https://arxiv.org/abs/2406.15643) | SIGGRAPH Asia 2024 | [humansensinglab/taming-3dgs](https://github.com/humansensinglab/taming-3dgs) | [Page](https://humansensinglab.github.io/taming-3dgs/) |
| [DashGaussian](https://arxiv.org/abs/2503.18402) | CVPR 2025 | [YouyuChen0207/DashGaussian](https://github.com/YouyuChen0207/DashGaussian) | [Page](https://dashgaussian.github.io/) |
| [Speedy-Splat](https://arxiv.org/abs/2412.00578) | CVPR 2025 | [j-alex-hanson/speedy-splat](https://github.com/j-alex-hanson/speedy-splat) | [Page](https://speedysplat.github.io/) |
| [FlashGS](https://arxiv.org/abs/2408.07967) | CVPR 2025 | [InternLandMark/FlashGS](https://github.com/InternLandMark/FlashGS) | — |
| [FastGS](https://arxiv.org/abs/2511.04283) | CVPR 2026 | [fastgs/FastGS](https://github.com/fastgs/FastGS) | [Page](https://fastgs.github.io/) |
| [Mini-Splatting2](https://arxiv.org/abs/2411.12788) | TPAMI 2026 | [fatPeter/mini-splatting2](https://github.com/fatPeter/mini-splatting2) | — |

---

<!-- ## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{fastergs2026,
  title     = {FasterGS: Workload-Balanced Gaussian Splatting for Faster Training and Rendering},
  author    = {Anonymous},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia (MM '26)},
  year      = {2026},
  note      = {Accepted}
}
``` -->

### Related baselines

```bibtex
@Article{kerbl3Dgaussians,
  author  = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  title   = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  journal = {ACM Transactions on Graphics},
  volume  = {42},
  number  = {4},
  month   = {July},
  year    = {2023},
  url     = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}

@inproceedings{fang2024minisplatting,
  title     = {Mini-Splatting: Representing Scenes with a Constrained Number of Gaussians},
  author    = {Fang, Guangchi and Wang, Bing},
  booktitle = {ECCV},
  year      = {2024}
}

@inproceedings{10.1145/3680528.3687694,
  author    = {Mallick, Saswat Subhajyoti and Goel, Rahul and Kerbl, Bernhard and Steinberger, Markus and Carrasco, Francisco Vicente and De La Torre, Fernando},
  title     = {Taming 3DGS: High-Quality Radiance Fields with Limited Resources},
  year      = {2024},
  booktitle = {SIGGRAPH Asia 2024 Conference Papers},
  articleno = {2},
  publisher = {ACM},
  doi       = {10.1145/3680528.3687694},
  url       = {https://doi.org/10.1145/3680528.3687694}
}

@InProceedings{Chen_2025_CVPR,
  author    = {Chen, Youyu and Jiang, Junjun and Jiang, Kui and Tang, Xiao and Li, Zhihao and Liu, Xianming and Nie, Yinyu},
  title     = {DashGaussian: Optimizing 3D Gaussian Splatting in 200 Seconds},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2025},
  pages     = {11146-11155}
}

@InProceedings{HansonSpeedy,
  author    = {Hanson, Alex and Tu, Allen and Lin, Geng and Singla, Vasu and Zwicker, Matthias and Goldstein, Tom},
  title     = {Speedy-Splat: Fast 3D Gaussian Splatting with Sparse Pixels and Sparse Primitives},
  booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
  month     = {June},
  year      = {2025},
  pages     = {21537-21546},
  url       = {https://speedysplat.github.io/}
}

@InProceedings{Feng_2025_CVPR,
  author    = {Feng, Guofeng and Chen, Siyan and Fu, Rong and Liao, Zimu and Wang, Yi and Liu, Tao and Hu, Boni and Xu, Linning and Pei, Zhilin and Li, Hengjie and Li, Xiuhong and Sun, Ninghui and Zhang, Xingcheng and Dai, Bo},
  title     = {FlashGS: Efficient 3D Gaussian Splatting for Large-scale and High-resolution Rendering},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2025},
  pages     = {26652-26662}
}

@InProceedings{Ren_2026_CVPR,
  author    = {Ren, Shiwei and Wen, Tianci and Fang, Yongchun and Lu, Biao},
  title     = {FastGS: Training 3D Gaussian Splatting in 100 Seconds},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2026},
  pages     = {26094-26103}
}

@article{fang2026minisplatting2,
  title   = {Efficient Scene Modeling via Structure-Aware and Region-Prioritized 3D Gaussians},
  author  = {Fang, Guangchi and Wang, Bing},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume  = {48},
  number  = {4},
  pages   = {4623--4641},
  year    = {2026},
  doi     = {10.1109/TPAMI.2025.3646473}
}

@InProceedings{Barron_2022_CVPR,
  author    = {Barron, Jonathan T. and Mildenhall, Ben and Verbin, Dor and Srinivasan, Pratul P. and Hedman, Peter},
  title     = {Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022},
  pages     = {5470-5479}
}
```

---

## License

Baseline folders retain their original licenses (typically research / non-commercial use under the GRAPHDECO 3DGS terms or the respective authors’ licenses). Please respect each upstream project’s LICENSE when redistributing or using commercially.

