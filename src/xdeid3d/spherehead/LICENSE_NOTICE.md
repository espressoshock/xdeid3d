# License Notice for SphereHead Module

This module (`xdeid3d.spherehead`) contains code adapted from multiple sources
with different licenses. Please review the following attributions carefully.

## Original Sources

### EG3D - NVIDIA Source Code License
The neural network architectures, volume rendering pipeline, and custom CUDA
operations are based on EG3D by NVIDIA:

- **Repository**: https://github.com/NVlabs/eg3d
- **Paper**: "Efficient Geometry-aware 3D Generative Adversarial Networks"
- **License**: NVIDIA Source Code License (NVIDIA proprietary)
- **Files affected**:
  - `torch_utils/` - PyTorch utilities and custom ops
  - `training/networks_stylegan2.py` - StyleGAN2 backbone
  - `training/networks_stylegan3.py` - StyleGAN3 improvements
  - `training/volumetric_rendering/` - Volume rendering pipeline
  - `dnnlib/` - Deep learning utilities

### SphereHead - Research License
The spherical tri-plane representation and modifications are from SphereHead:

- **Repository**: https://github.com/lhyfst/spherehead
- **Paper**: "SphereHead: Stable 3D Full-head Synthesis with Spherical Tri-plane Representation"
- **Authors**: Heyuan Li, Ce Chen, Tianhao Shi, Yuda Qiu, Sizhe An, Guanying Chen, Xiaoguang Han
- **License**: Research/Academic use
- **Files affected**:
  - `training/triplane.py` - Spherical tri-plane generator
  - `camera_utils.py` - Camera pose sampling

## Usage Restrictions

The code in this module is provided for **research and educational purposes only**.
Commercial use may require additional licensing from:
1. NVIDIA Corporation for EG3D-derived code
2. The SphereHead authors for spherical tri-plane modifications

## X-DeID3D Integration

The integration layer and modifications made for X-DeID3D are licensed under
Apache-2.0. This includes:
- Import path restructuring
- Package organization
- Documentation and examples
- Integration with the X-DeID3D evaluation framework

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{li2024spherehead,
    title={SphereHead: Stable 3D Full-head Synthesis with Spherical Tri-plane Representation},
    author={Heyuan Li and Ce Chen and Tianhao Shi and Yuda Qiu and Sizhe An and Guanying Chen and Xiaoguang Han},
    year={2024},
    eprint={2404.05680},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@inproceedings{chan2022eg3d,
    title={Efficient Geometry-aware 3D Generative Adversarial Networks},
    author={Chan, Eric R. and Lin, Connor Z. and Chan, Matthew A. and Nagano, Koki and Pan, Boxiao and De Mello, Shalini and Gallo, Orazio and Guibas, Leonidas and Tremblay, Jonathan and Khamis, Sameh and Karras, Tero and Wetzstein, Gordon},
    booktitle={CVPR},
    year={2022}
}
```
