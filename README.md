# DMSAN
The official PyTorch implementation of Deep Mixed Subdomain Adaptation Networks
## Requirement
- PyTorch  
- Pillow can't be 8.3.0 


## Useage
1. Run `makedir.py` to generate directories. 
2. Put  the unzipped datasets into the folder "dataset".
3. Run `DMSAN.py` to training models.

## Datasets
- The cross-species plant disease severity dataset(CSPDS) ( [GoogleDrive](https://drive.google.com/drive/folders/1r94_8BkUpdREsfUyCl0jWN8Lbf1-TwC1?usp=sharing))


## License
The source code is free for research and education use only. Any comercial use should get formal permission first.
Part of code comes from this work:
```
@article{zhu2020deep,
  title={Deep Subdomain Adaptation Network for Image Classification},
  author={Zhu, Yongchun and Zhuang, Fuzhen and Wang, Jindong and Ke, Guolin and Chen, Jingwu and Bian, Jiang and Xiong, Hui and He, Qing},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2020},
  publisher={IEEE}
}
```
Our work：
```
@article{yan2021deep,
  title={Deep Transfer Learning for Cross-Species Plant Disease Diagnosis Adapting Mixed Subdomains},
  author={Yan, Ke and Guo, Xinlu and Ji, Zhiwei and Zhou, Xiaokang},
  journal={IEEE/ACM transactions on computational biology and bioinformatics},
  year={2021},
  publisher={IEEE}
}
```
