# LipstickFinder-Backend
## Introduction
The backend of Lipstick Finder has 4 major functions:

*	Lipstick recognition
*	Lip digital makeup
*	Daily lipsticks recommendation
*	User management
## Directory Tree
``` bash
│  app.py (Start our Flask server)
│  requirements.txt (Includes the required packages)
│  
├─models
│      model.py (BiSeNet model)
│      resnet.py (ResNet model)
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

## References
Code Source: https://github.com/zllrunning/face-parsing.PyTorch
