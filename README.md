# Transferable Neural Networks for Partial Differential Equations
Code repository for the paper:  
**Transferable Neural Networks for Partial Differential Equations** <br>
[Zezhong Zhang](https://www.ornl.gov/staff-profile/zezhong-zhang), [Feng Bao](https://www.math.fsu.edu/~bao/), [Lili Ju](https://people.math.sc.edu/ju), [Guannan Zhang](https://sites.google.com/view/guannan-zhang) <br>
Journal of Scientific Computing, 2024 <br>
[[paper](https://link.springer.com/article/10.1007/s10915-024-02463-y)]


- PDE accuracy: First run "PDE_(problem_name)/problem_setup.ipynb" to generate true solution data. Then run "PDE_(problem_name)/ls.ipynb" for LS-based solutions (TransNet and random feature model), and "PDE_(problem_name)/pinn.ipynb" for PINN.
- Shape parameter tuning: See "basis_train_2d/." and "basis_train_3d/.".
- Hyperplane density simulation: Simulated hyperplane density can be found in "basis_analysis/density_plot.ipynb".
- Portable scripts (for Possion 2D and NS steady state): Single scripts that do not depend on the rest of the repo. 


## Citation
If you  find the idea or code of this paper useful for your research, please consider citing us:

```bibtex
@article{zhang2024transferable,
  title={Transferable Neural Networks for Partial Differential Equations},
  author={Zhang, Zezhong and Bao, Feng and Ju, Lili and Zhang, Guannan},
  journal={Journal of Scientific Computing},
  volume={99},
  number={1},
  pages={2},
  year={2024},
  publisher={Springer}
}
```
