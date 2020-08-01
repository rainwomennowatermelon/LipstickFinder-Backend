# LipstickFinder-Backend
## Introduction
The backend of Lipstick Finder has 4 major functions:

*	Lipstick recognition
*	Lip digital makeup
*	Daily lipsticks recommendation
*	User management

The *evaluation*  folder is an independent module. It gives test to the baseline and RMBD algorithm on our test dataset.
## Directory Tree
### Backend
``` bash
│  app.py (Start our Flask server)
│  requirements.txt (Includes the required packages)
│  
├─models
│      model.py (BiSeNet model)
│      resnet.py (ResNet-18 model)
│      
├─res
│  ├─cp
│  │      79999_iter.pth (Pre-trained face-parsing model)
│  │      
│  └─data
│      ├─face-parsing-makeupImgDir (Lip makeup images store path)
│      ├─face-parsing-predictImgDir (Lipstick recognition images store path)
│      └─profiles (Profile images store path)
│              
├─scheduledJob
│      userBasedCF.py (Daily job to generate collaborative filtering result)
│      
├─src
│      lipstickRecommendation.py (Lipstick recommendation SDK)
│      usersManagement.py (User management SDK)
│      
└─utils
        colorMethods.py (Color transformation util)
```

### Evaluation
``` bash
│  testBaselineProgram.py (The baseline's test program: without points removement)
│  testRmbdProgram.py (RMBD's test program)
│  
├─json
│      lipsticksMod.json (Our updated color database)
│      
├─models
│      model.py (BiSeNet model)
│      resnet.py (ResNet-18 model)
│      
└─testDataset
    ├─daily (Labeled lipsticks in daily life style)
    └─dior
        ├─rouge (Dior rouge style)
        ├─rougered (Dior rouge red style)
        ├─seductive (Dior seductive style)
        └─star (Dior star style)
```
## References
Code Source: https://github.com/zllrunning/face-parsing.PyTorch
