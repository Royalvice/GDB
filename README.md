# GDB: Gated convolutions-based Document Binarization
## Description
This is an official implementation for the paper [GDB: Gated convolutions-based Document Binarization](https://arxiv.org/abs/2302.02073).

This repository also comprehensively collects the datasets that may be used in document binarization.
## Datasets

Below is a summary table of the datasets used for document binarization, along with links to download them.

## Environment

- Python >= 3.7
- torch >= 1.7.0
- torchvision >= 0.8.0

## Usage

### Prepare the dataset

**Note**: The pre-processing code is not provided yet. But it is on the way.

You can download the datasets from the links below and put them in the `datasets_ori` folder. 
When evaluating performance on the DIBCO2019 dataset, 
first gather all datasets except for DIBCO2019 and place them in the `img` and `gt` folders under the `datasets_ori` directory.
Then crop the images and ground truth images into patches (256 * 256) and place them in the `img` and `gt` folders under the `datasets/DIBCO2019` directory.
Next, use the Otsu thresholding method to binaryze the images 
under `datasets/img` and place the results in the `datasets/otsu` folder. 
Use the Sobel operator to process the images under `datasets/img` 
and place the results in the `datasets/sobel` folder. 
With these preprocessing steps completed, 
Pass `./datasets/img` as an argument for the `--dataRoot` parameter in `train.py` and begin training.
### Training

```shell
python train.py
```

### Testing

```shell
python test.py
```

## Datasets

| Dataset                                                                                                                                                      |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [DIBCO 2009](http://users.iit.demokritos.gr/~bgat/DIBCO2009/benchmark/)                                                                                      |
| [H-DIBCO 2010](http://users.iit.demokritos.gr/~bgat/H-DIBCO2010/benchmark/)                                                                                  |
| [DIBCO 2011](http://utopia.duth.gr/~ipratika/DIBCO2011/benchmark/)                                                                                           |
| [H-DIBCO 2012](http://utopia.duth.gr/~ipratika/HDIBCO2012/benchmark/)                                                                                        |
| [DIBCO 2013](http://utopia.duth.gr/~ipratika/DIBCO2013/benchmark/)                                                                                           |
| [H-DIBCO 2014](http://users.iit.demokritos.gr/~bgat/HDIBCO2014/benchmark/)                                                                                   |
| [H-DIBCO 2016](http://vc.ee.duth.gr/h-dibco2016/benchmark/)                                                                                                  |
| [DIBCO 2017](http://vc.ee.duth.gr/dibco2017/benchmark/)                                                                                                      |
| [H-DIBCO 2018](https://vc.ee.duth.gr/h-dibco2018/benchmark/)                                                                                                 |
| [DIBCO 2019](https://vc.ee.duth.gr/dibco2019/benchmark/)                                                                                                     |
| [Palm Leaf Manuscript](http://amadi.univ-lr.fr/ICFHR2016_Contest/)                                                                                           |
| [Persian Heritage Image Binarization Dataset (PHIBD)](http://www.iapr-tc11.org/mediawiki/index.php/Persian_Heritage_Image_Binarization_Dataset_(PHIBD_2012)) |
| [Ensiedeln](http://www.e-codices.unifr.ch/en/sbe/0611/)                                                                                                      |
| [Noisy Office](https://archive.ics.uci.edu/ml/datasets/NoisyOffice)                                                                                          |
| [Synchromedia Multispectral dataset](https://tc11.cvc.uab.es/datasets/SMADI_1)                                                                               |
| [Bickly-diary dataset](https://github.com/vqnhat/DSN-Binarization/files/2793688/original_gt_labeled.zip)                                                     |
| [IAM Historical Document Database](https://fki.tic.heia-fr.ch/databases/iam-historical-document-database)                                                    |


## To-do list
- [x] Add the code for training
- [x] Add the code for testing
- [ ] Add the code for pre-processing
- [ ] Restruct the code
- [ ] Upload the pretrained weights
- [x] Comprehensively collate document binarization benchmark datasets
- [ ] Add the code for evaluating the performance of the model
## License
This work is permitted for academic research purposes only. For commercial use, please contact the author. 
## Citation
- If this work is useful, please cite it as: 
```
@article{yang2024gdb,
  title={GDB: gated convolutions-based document binarization},
  author={Yang, Zongyuan and Liu, Baolin and Xiong, Yongping and Wu, Guibin},
  journal={Pattern Recognition},
  volume={146},
  pages={109989},
  year={2024},
  publisher={Elsevier}
}
```

