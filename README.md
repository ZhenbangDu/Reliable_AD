# Towards Reliable Advertising Image Generation Using Human Feedback [ECCV2024]
The official code and dataset for the paper Towards Reliable Advertising Image Generation Using Human Feedback (ECCV2024)
- Authors: Zhenbang Du, Wei Feng, Haohan Wang, Yaoyu Li, Jingsen Wang, Jian Li, Zheng Zhang, Jingjing Lv, Xin Zhu, Junsheng Jin, Junjie Shen, Zhangang Lin, and Jingping Shao

<img width="928" alt="image" src="figures/Figure1.png">  

## Abstract
In the e-commerce realm, compelling advertising images are pivotal for attracting customer attention. While generative models automate image generation, they often produce substandard images that may mislead customers and require significant labor costs to inspect. This paper delves into increasing the rate of available generated images. We first introduce a multi-modal Reliable Feedback Network (RFNet) to automatically inspect the generated images. Combining the RFNet into a recurrent process, Recurrent Generation, results in a higher number of available advertising images. To further enhance production efficiency, we fine-tune diffusion models with an innovative Consistent Condition regularization utilizing the feedback from RFNet (RFFT). This results in a remarkable increase in the available rate of generated images, reducing the number of attempts in Recurrent Generation, and providing a highly efficient production process without sacrificing visual appeal. We also construct a Reliable Feedback 1 Million (RF1M) dataset which comprises over one million generated advertising images annotated by humans, which helps to train RFNet to accurately assess the availability of generated images and faithfully reflect the human feedback. Generally speaking, our approach offers a reliable solution for advertising image generation.

## RF1M Dataset
[TrainSet](https://3.cn/-10gOQ79s)

[TestSet](https://3.cn/10gO8kw-K)


## Getting started

- Python >= 3.9 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.13.0+cu11.7](https://pytorch.org/)
```bash
conda create -n reliable python==3.9.0
conda activate reliable
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```
## How to inference
```bash
cd Reliable_AD
python sample.py \
--base_model_path "digiplay/majicMIX_realistic_v7" \
--controlnet_model_path  "lllyasviel/control_v11p_sd15_canny" \
--batch_size 10 \
--sampler_name 'DDIM'  \
--num_inference_steps 40 \
--config ./config/config.json  \
--data_path  ./samples \
--save_path ./outputs
```
## Citation
```
@inproceedings{du2024reliablead,
    title={Towards Reliable Advertising Image Generation Using Human Feedback},
    author={Zhenbang, Du and Wei, Feng and Haohan, Wang and Yaoyu, Li and Jingsen, Wang and Jian, Li and Zheng, Zhang and Jingjing, Lv and Xin, Zhu and Junsheng, Jin and Junjie, Shen and Zhangang, Lin and Jingping, Shao},
    booktitle={European Conference on Computer Vision},
    year={2024},
}
```
