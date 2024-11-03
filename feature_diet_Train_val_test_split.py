import argparse
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import os



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='/path', type=str, help='path to the label file')
    parser.add_argument('--split', default=0.2, type=float, help='Train/Test split ratio')
    parser.add_argument('--folds', default=5, type=int, help='No of folds in K-folds')
    args = parser.parse_args()

    # loading the annotation file :-
    df = pd.read_csv(args.dataset)
    df = df.sample(frac=1).reset_index(drop=True)

    path_train, path_test, label_8_way_train, label_8_way_test, label_2_way_train, label_2_way_test  = train_test_split(df['path'], df['label_8_way'], df['label_2_way'],   test_size = args.split, stratify=df['label_8_way'], random_state=42)
    path_train = path_train.to_frame()
    path_test = path_test.to_frame()
    label_8_way_train = label_8_way_train.to_frame()
    label_8_way_test = label_8_way_test.to_frame()
    label_2_way_train = label_2_way_train.to_frame()
    label_2_way_test = label_2_way_test.to_frame()

    df_train = path_train.join(label_8_way_train).join(label_2_way_train).reset_index(drop=True)
    df_test = path_test.join(label_8_way_test).join(label_2_way_test).reset_index(drop=True)

    path = '/kaggle/working/'
    os.makedirs(os.path.join(path, 'Test'), exist_ok=True)
    df_test.to_csv(os.path.join(path, 'Test', 'test.csv'), index=False)


    kf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    i=0
    os.makedirs(os.path.join(path, 'Train_Val_split'), exist_ok=True)


    for train_idx, val_idx in kf.split(df_train['path'], df_train['label_8_way'], df_train['label_2_way']):
        train_df = df_train.iloc[train_idx].reset_index(drop=True)
        train = str(i) + 'train.csv'
        train_df.to_csv(os.path.join(path, 'Train_Val_split', train), index=False)

        df_val = df_train.iloc[val_idx].reset_index(drop=True)
        val = str(i) + 'val.csv'
        df_val.to_csv(os.path.join(path, 'Train_Val_split', val), index=False)
        i = i + 1





if __name__ == '__main__':
    main()