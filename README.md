<img src="assets/logo.webp" width="100%" align="center">
<h1 align="center">A Simple dynamic extension for TRELLIS</h1>

- This code aims to provide a training-free strategy for TRELLIS to generate continuous results through time. 
- This is a forked version from TRELLIS. For the original work, please refer to the links below.

<p align="center"><a href="https://arxiv.org/abs/2412.01506"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
<a href='https://trellis3d.github.io'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
<a href='https://huggingface.co/spaces/JeffreyXiang/TRELLIS'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live_Demo-blue'></a>
</p>

## Example Result
<p align="center">
    <img src="assets/spiderman_mesh_thumbnail.png" width="240" height="240" alt="Spiderman Mesh">
    <img src="assets/spiderman_mesh_original_thumbnail.png" width="240" height="240" alt="Spiderman Mesh Original">
</p>
Mesh result becomes obviously better in this algorithm. The overall shape is controlled in a stable shape through time.

<p align="center">
    <img src="assets/spiderman_gs_thumbnail.png" width="240" height="240" alt="Spiderman GS">
    <img src="assets/spiderman_gs_original_thumbnail.png" width="240" height="240" alt="Spiderman GS Original">
</p>
The GS results have also improved, but there are still occasional flickers in the algorithm.

<!-- Installation -->
## 📦 Installation 

Please follow the original installation in TRELLIS.

## 🔨 Requirement
System RAM 48G at least, CUDA RAM 10G at least.
The strategy used in this repo is very System-RAM consuming, as it saves all the attention to CPU. You can delete the `to('cpu')` in `trellis/modules/sparse/attention/modules.py` and `trellis/modules/attention/modules.py` to exchange System RAM for CUDA RAM if you have a better GPU.

<!-- Usage -->
## 💡 Usage
1. Place all the images under a directory, make sure their names are correct (e.g., 00.png-99.png).
2. Run `dynamic.py` or `dynamic_cpu_efficient.py` according to your System RAM.
3. See the result in output. It saves all the frames' results and a final result combining all frames.

## Other Cases
<p align="center">
    <img src="assets/ironman_gs_thumbnail.png" width="240" height="240" alt="Ironman GS">
    <img src="assets/ironman_mesh_thumbnail.png" width="240" height="240" alt="Ironman Mesh">
</p>
<p align="center">
    <img src="assets/pistol_gs_thumbnail.png" width="240" height="240" alt="Pistol GS">
    <img src="assets/pistol_mesh_thumbnail.png" width="240" height="240" alt="Pistol Mesh">
</p>

## Failure Case
<p align="center">
    <img src="assets/robot_mesh_thumbnail.png" width="240" height="240" alt="Robot Mesh">
</p>

<!-- License -->
## ⚖️ License

TRELLIS models and the majority of the code are licensed under the [MIT License](LICENSE). The following submodules may have different licenses:
- [**diffoctreerast**](https://github.com/JeffreyXiang/diffoctreerast): We developed a CUDA-based real-time differentiable octree renderer for rendering radiance fields as part of this project. This renderer is derived from the [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization) project and is available under the [LICENSE](https://github.com/JeffreyXiang/diffoctreerast/blob/master/LICENSE).

- [**Modified Flexicubes**](https://github.com/MaxtirError/FlexiCubes): In this project, we used a modified version of [Flexicubes](https://github.com/nv-tlabs/FlexiCubes) to support vertex attributes. This modified version is licensed under the [LICENSE](https://github.com/nv-tlabs/FlexiCubes/blob/main/LICENSE.txt).

<!-- Citation -->
## 📜 Citation

If you find this work helpful, please consider citing the original paper:

```bibtex
@article{xiang2024structured,
    title   = {Structured 3D Latents for Scalable and Versatile 3D Generation},
    author  = {Xiang, Jianfeng and Lv, Zelong and Xu, Sicheng and Deng, Yu and Wang, Ruicheng and Zhang, Bowen and Chen, Dong and Tong, Xin and Yang, Jiaolong},
    journal = {arXiv preprint arXiv:2412.01506},
    year    = {2024}
}
```
