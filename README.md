# SOMGAN -- Official PyTorch implementation

"Improving Mode Exploring Capability of Generative Adversarial Nets by Self-Organizing Map"

![docs/architecture.png](docs/architecture.png)

In this paper, we propose a new approach to train the GANs with one generator and a mixture of discriminators to overcome the mode collapse problem. 
In our model, each discriminator not only distinguishes real and fake samples but also differentiates modes in datasets. 
In essence, it combines a classical clustering idea called Self-Organizing Map and multiple discriminators into a unified optimization objective.
Specifically, we define a topological structure over the multiple discriminators in order to diversify the generated samples and to capture multi-modes. 
We term this method Self-Organizing Map Generative Adversarial Nets (SOMGAN).
By utilizing the parameter sharing trick, the proposed model requires trivial extra computation compared with GANs with a single discriminator. 
In our experiment, the method covers diverse data modes and gives outstanding performance in qualitative and quantitative evaluations. 
Since the topological constraint of discriminators is irrelevant to the generator, the SOM-based framework can be embedded into arbitrary GAN frameworks to maximize the generative capacity of the target model.



## datasets
| Dataset      | #images | Size |  Link
| ----------- | ----------- | ----------- | ----------- |
| CIFAR-10      | 50,000       |158.3 MB| <a href="https://drive.google.com/file/d/1QhWFcQJtp1wmZUcN7AVgUliDz03C9gOc/view?usp=sharing"><img src="./docs/GoogleDrive.svg" height="45" alt="Google Drive Datasets"></a>|
| STL-10   | 100,000        |2.6 GB| <a href="https://drive.google.com/file/d/1FgbU6_kaJLlJr5TIultCxcX1apOBkemH/view?usp=sharing"><img src="./docs/GoogleDrive.svg" height="45" alt="Google Drive Datasets"></a>
|CelebA | 202,599 | 2.37 GB| <a href="https://drive.google.com/file/d/1j26kEFYjMOBqY4y2s5Pe6FTOWF5UKNiB/view?usp=sharing"><img src="./docs/GoogleDrive.svg" height="45" alt="Google Drive Datasets"></a>





## metrics: FID

Linear convolutional GAN architectures:

 Models                            | CIFAR-10             |      STL-10          | CelebA         
-----------------------------------------|------------------:|----------------:|---------------:
 Dropout-GAN                             | 88.6             |                | 36.36         
 DCGAN                                   | 37.7             |                
 DCGAN+TTUR                              | 36.9             |                
 WGAN-GP }                               | 29.3             | 55.1           
 SNGAN (linear G + linear D)       | 29.3             | 53.1           
 MGAN                                    | 26.7             |                
 SOMGAN 9D (linear G + linear D)  | **20.48**            | **56.19**          | **8.46**          
 SOMGAN 25D (linear G + linear D) | **19.88**     | **52.76** | **5.93** 
 
 ResNet convolutional GAN architectures:
 
 Models                            | CIFAR-10             |      STL-10          | CelebA         
-----------------------------------------|------------------:|----------------:|---------------:
 WGAN-GP+TTUR                            | 24.8             |                
 SNGAN (ResNet G + skip D)         | 21.7             | 40.1           
 PeerGAN                                 | 21.55            | 51.37          | 13.95         
 BigGAN                                  | 14.73            |                
 Autogan                                 | 12.42   | 31.01          |               
 SOMGAN 9D (ResNet G + skip D)    | **13.51**            | **30.12** | **5.65** 
 
 Style convolutional GAN architectures:
 
 Models                            | CIFAR-10             |      STL-10          | CelebA         
-----------------------------------------|------------------:|----------------:|---------------:
 StyleGAN2                               | 11.07            |                | 5.06          
 LT-GAN                                  | 9.80             | 31.35          | 16.84         
 SOMGAN 4D (style G + linear D)   | **3.05**   | **24.49** | **2.89** 


## training files of the best records

1. best model kpl file
2. training logs
3. FID metric history 
4. training options
5. loss status logs
6. fake image examples

SOMGAN 4D (style G + linear D) | CIFAR-10             |      STL-10          | CelebA         
-----------------------------------------|------------------:|----------------:|---------------:
FID | **3.05**   | **24.49** | **2.89** 
Link|<a href="https://drive.google.com/drive/folders/1ctHztASVZEIH_-Pmh8TZ2SjuL5bwVDtm?usp=sharing"><img src="./docs/GoogleDrive.svg" height="45" alt="Google Drive Datasets"></a>|<a href="https://drive.google.com/drive/folders/1shp88i7SD-ojS0mydmoGgiJm9kYvSAn8?usp=sharing"><img src="./docs/GoogleDrive.svg" height="45" alt="Google Drive Datasets"></a>|<a href="https://drive.google.com/drive/folders/12lieiS1Y3rna6aDNbVIzW4fLtV15oPXN?usp=sharing"><img src="./docs/GoogleDrive.svg" height="45" alt="Google Drive Datasets"></a>|

## Generate examples

```.bash
# Generate images using best pretrained_weight 
python generate.py --outdir=out --seeds=1-100 --network=path_to_pkl_file
```
