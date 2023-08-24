import csv
import os
import pandas as pd  
import torch
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.utils import compute_class_weight

# creating empty DataFrame
train_df = pd.DataFrame() 
val_df = pd.DataFrame() 
test_df = pd.DataFrame() 
camera_names = ['basler', 'webcam_c', 'webcam_l', 'webcam_r' ]
eve_dataset = 'eve/src/hope/images_face'
d_train=[]
d_test=[]
d_val = []

for dirs in sorted(os.listdir(eve_dataset)):
    if dirs.startswith('train'):
        for stimuli in sorted(os.listdir(os.path.join(eve_dataset, dirs))):
            
            for step in sorted(os.listdir(os.path.join(eve_dataset, dirs, stimuli))):

                for cameras in sorted(os.listdir(os.path.join(eve_dataset, dirs, stimuli, step))):
                    if cameras == camera_names[1]:
                        img = [os.path.splitext(filename)[0] for filename in sorted(os.listdir(os.path.join(eve_dataset, dirs, stimuli, step, cameras)))]
                        img= sorted(img, key=int)
                        img= img[:30]
                        for images in sorted(img, key=int):
                            extension = '.png'
                            imgs= images+extension
                            d_train.append(
                                {
                                    'path': os.path.join('images_face', dirs, stimuli,step, camera_names[1]),
                                    'label': stimuli
                                }
                            )
    elif dirs.startswith('val'):
        for stimuli in sorted(os.listdir(os.path.join(eve_dataset, dirs))):
            for step in sorted(os.listdir(os.path.join(eve_dataset, dirs, stimuli))):
                for cameras in sorted(os.listdir(os.path.join(eve_dataset, dirs, stimuli, step))):
                    if cameras == camera_names[1]:
                        img = [os.path.splitext(filename)[0] for filename in sorted(os.listdir(os.path.join(eve_dataset, dirs, stimuli, step, cameras)))]
                        img= sorted(img, key=int)
                        img= img[:30]
                
                        for imgs in sorted(img, key=int):
                            
                            extension = '.png'
                            imgs= imgs+extension
                            d_val.append(
                                {
                                    'path': os.path.join('images_face', dirs, stimuli,step, camera_names[1]),
                                    'label': stimuli
                                }
                            )
    elif dirs.startswith('test'):
        for stimuli in sorted(os.listdir(os.path.join(eve_dataset, dirs))):
            for step in sorted(os.listdir(os.path.join(eve_dataset, dirs, stimuli))):
                for cameras in sorted(os.listdir(os.path.join(eve_dataset, dirs, stimuli, step))):
                    if cameras == camera_names[1]:
                        img = [os.path.splitext(filename)[0] for filename in sorted(os.listdir(os.path.join(eve_dataset, dirs, stimuli, step, cameras)))]
                        img= sorted(img, key=int)
                        img= img[:30]
                        for imgs in sorted(img, key=int):
                            extension = '.png'
                            imgs= imgs+extension
                            d_test.append(
                                {
                                    'path': os.path.join('images_face', dirs, stimuli,step, camera_names[1]),
                                    'label': stimuli
                                }
                            )

train_df = pd.DataFrame(d_train)

val_df = pd.DataFrame(d_val)

test_df = pd.DataFrame(d_test)

# path_label_train = train_df.to_csv('data_train.csv', index=False)
# path_label_val = val_df.to_csv('data_val.csv', index=False)
# path_label_test = test_df.to_csv('data_test.csv', index=False)


class_weights = compute_class_weight(
                                    class_weight = "balanced",
                                    classes = np.unique(train_df['label']),
                                    y = train_df['label']                                                    
                                )
#class_weights = dict(zip(np.unique(targets), class_weights)) 
print(class_weights)
