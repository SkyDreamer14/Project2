# Generated_Image_Classification

### I. Project introduction
- __Project goal__  
  - To develop a deep learning model that effectively distinguishes between real and AI-generated images  
  - The model helps prevent the misuse of generative models (e.g., Stable Diffusion) by detecting fake content before it can be exploited  
 ![image](https://github.com/user-attachments/assets/99ea347b-5692-43ad-b086-d97eae6abd32)
- __Project motivation__
  - Preventing misinformation and manipulation   
    - Malicious users can exploit generative models to create photorealistic fake images.  
    - These images can be used to:  
      Fabricate events, harass individuals, spread propaganda or deepfakes  
  - Real-world impact of synthetic images  
    - When fake images are mistaken for real ones in social media, news, or legal contexts, it can:  
    - Undermine public trust, spread confusion and hate, cause reputational and psychological harm  
  - Need for explainable AI  
    - As generative models evolve, detecting fake images becomes more difficult.  
    - This project not only classifies images, but also uses Grad-CAM to visualize what the model is focusing on enhancing transparency.  

### II. Dataset Description  
- __The dataset consists of two categories__
  - Two categories of real images and AI-generated fake images (generated using Stable Diffusion, a text-to-image diffusion model).
  - The dataset includes a total of 4000 images, with 2000 real images and 2000 fake images.


- __Data splitting strategy__
  - Training dataset:   
    - Contains 80% of the total dataset (3200 images)  
    - Includes both real and fake images with corresponding labels (real or fake)  
  - Validation dataset:    
    - Contains 20% of the total dataset (800 images)  
    - Used to evaluate model generalization and select the best model  
      ![image](https://github.com/user-attachments/assets/22188b1c-25e2-49ff-a83f-2599be46cd44)
  - **You can get the file from this link** https://github.com/hellominjeong/Generated_Image_Classification.

  ### III. Baseline model
  - __Classification model__  
    - Pre-trained EfficientNet-B0  
    - Optimizer: Adam  
    - Loss function: CrossEntropyLoss  
    - Training hyperparameters:  
      - Batch size: 32  
      - Epochs: 10  
      - Learning rate: 0.003  
  - __Training and validation datasets transforms__  
    - Training dataset:     
      - Resize to (224, 224)    
      - RandomHorizontalFlip (data augmentation  
      - Normalize: (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    - Validation dataset:      
      - Resize to (224, 224)   
      - Normalize: (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      
  - __Model train and evaludation__
    - Training accuracy: 0.9869
    - Training loss: 0.0351
    - Best validation accuracy: 0.9962
     ![image](https://github.com/user-attachments/assets/93e585ec-7582-41c5-a380-be17a35cb0b1)  

    
    



