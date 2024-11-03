import argparse
import os
from dataset import feature_extraction_dataset 
import torch
from torch.utils.data import DataLoader
from model import Feature_Extractor_Diet
import numpy as np
import pandas as pd



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', default=8, type=int, help='2:- Normal vs Abnormal')
    parser.add_argument('--file', default='/path', type=str, help='path to the csv file ')
    parser.add_argument('--model', default='Feature_Extractor_Diet', type=str, help='Model to be used for feature extraction')
    parser.add_argument('--batch_size', default=512, type=int, help='Batch size for the dataloader')
    parser.add_argument('--num_workers', default=4, type=int, help='num workers for dataloader')
    args = parser.parse_args()

    dataset = feature_extraction_dataset(df_file = args.file)
    data_loader = DataLoader(dataset, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)



    if args.model == 'Feature_Extractor_Diet':
        model = Feature_Extractor_Diet()
        device = torch.device("cuda")
        model.to(device)



    features_list = []
    path_list = []
    label_2_way_list = []
    label_8_way_list = []
    i = 0
    for data in data_loader:
        i += 1
        print(i)
        feature = data[0].float()
        img_path = data[1]
        label_8_way = data[2]
        label_2_way = data[3]

         # Check if CUDA is available
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")


        # Move the tensor to the selected device (CPU or CUDA)
        feature = feature.to(device)

        img_features = model(feature)

        features_list.extend(img_features.detach().cpu())
        path_list.extend(img_path)
        label_8_way_list.extend(label_8_way.detach().cpu().tolist())
        label_2_way_list.extend(label_2_way.detach().cpu().tolist())

        torch.cuda.empty_cache()

        


   
    numpy_array = np.stack([t.numpy() for t in features_list])
    df_features = pd.DataFrame(numpy_array)

    df_path = pd.DataFrame(path_list, columns=['path'])

    df_label_8_way = pd.DataFrame(label_8_way_list, columns=['label_8_way'])

    df_label_2_way = pd.DataFrame(label_2_way_list, columns=['label_2_way'])

    df_combined = pd.concat([df_path, df_label_8_way, df_label_2_way, df_features], axis=1)


    path = '/kaggle/working/'
    output = 'features_Histopath_Breast_cancer.csv'
    df_combined.to_csv(os.path.join(path, output), index=False)






if __name__ == '__main__':
    main()