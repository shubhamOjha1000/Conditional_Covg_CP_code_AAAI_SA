import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from CP_methods import THR, APS, RAPS
import torch
from utils import avg_set_size_metric, coverage_gap_metric
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Trials', default=100, type=int, help= 'Number of total trials')
    parser.add_argument('--softmax_output_file_path', default='/path', type=str, help='path to the softmax_output_file')
    parser.add_argument('--expt_no', default=3, type=int, help= 'Expt no :-1, 2, 3, 4')
    parser.add_argument('--split', default=0.1, type=float, help='Calib/test split ratio')
    parser.add_argument('--CP_method', default='THR', type=str, help='CP method :- 1)THR  2)APS  3)RAPS')
    args = parser.parse_args()


    adenosis_avg_set_size_len_for_T_trials = []
    fibroadenoma_avg_set_size_len_for_T_trials = []
    phyllodes_tumor_avg_set_size_len_for_T_trials = []
    tubular_adenoma_avg_set_size_len_for_T_trials = []

    ductal_carcinoma_avg_set_size_len_for_T_trials = []
    lobular_carcinoma_avg_set_size_len_for_T_trials = []
    mucinous_carcinoma_avg_set_size_len_for_T_trials = []
    papillary_carcinoma_avg_set_size_len_for_T_trials = []




    adenosis_coverage_for_T_trials = []
    fibroadenoma_coverage_for_T_trials = []
    phyllodes_tumor_coverage_for_T_trials = []
    tubular_adenoma_coverage_for_T_trials = []

    ductal_carcinoma_coverage_for_T_trials = []
    lobular_carcinoma_coverage_for_T_trials = []
    mucinous_carcinoma_coverage_for_T_trials = []
    papillary_carcinoma_coverage_for_T_trials = []



    for t in range(args.Trials):
        print()
        print(f'Trials :- {t}')
        print()


        # loading the annotation file :-
        df = pd.read_csv(args.softmax_output_file_path)
        df = df.sample(frac=1).reset_index(drop=True)

        df_adenosis = df[df['Label'] == 0]
        df_fibroadenoma = df[df['Label'] == 1]
        df_phyllodes_tumor = df[df['Label'] == 2]
        df_tubular_adenoma = df[df['Label'] == 3]
        df_ductal_carcinoma = df[df['Label'] == 4]
        df_lobular_carcinoma = df[df['Label'] == 5]
        df_mucinous_carcinoma = df[df['Label'] == 6]
        df_papillary_carcinoma = df[df['Label'] == 7]



        #adenosis:- 
        df_features_test, df_features_claib, df_label_test, df_label_claib = train_test_split(df_adenosis.iloc[:, :8], df_adenosis['Label'], test_size = args.split, stratify=df_adenosis['Label'], random_state=42)
        df_adenosis_test = df_features_test.join(df_label_test, how='inner').reset_index(drop=True)
        df_adenosis_claib = df_features_claib.join(df_label_claib, how='inner').reset_index(drop=True)


        #fibroadenoma:-
        df_features_test, df_features_claib, df_label_test, df_label_claib = train_test_split(df_fibroadenoma.iloc[:, :8], df_fibroadenoma['Label'], test_size = args.split, stratify=df_fibroadenoma['Label'], random_state=42)
        df_fibroadenoma_test = df_features_test.join(df_label_test, how='inner').reset_index(drop=True)
        df_fibroadenoma_claib = df_features_claib.join(df_label_claib, how='inner').reset_index(drop=True)


        #phyllodes_tumor:- 
        df_features_test, df_features_claib, df_label_test, df_label_claib = train_test_split(df_phyllodes_tumor.iloc[:, :8], df_phyllodes_tumor['Label'], test_size = args.split, stratify=df_phyllodes_tumor['Label'], random_state=42)
        df_phyllodes_tumor_test = df_features_test.join(df_label_test, how='inner').reset_index(drop=True)
        df_phyllodes_tumor_claib = df_features_claib.join(df_label_claib, how='inner').reset_index(drop=True)


        #tubular_adenoma:-
        df_features_test, df_features_claib, df_label_test, df_label_claib = train_test_split(df_tubular_adenoma.iloc[:, :8], df_tubular_adenoma['Label'], test_size = args.split, stratify=df_tubular_adenoma['Label'], random_state=42)
        df_tubular_adenoma_test = df_features_test.join(df_label_test, how='inner').reset_index(drop=True)
        df_tubular_adenoma_claib = df_features_claib.join(df_label_claib, how='inner').reset_index(drop=True)


        #ductal_carcinoma:-
        df_features_test, df_features_claib, df_label_test, df_label_claib = train_test_split(df_ductal_carcinoma.iloc[:, :8], df_ductal_carcinoma['Label'], test_size = args.split, stratify=df_ductal_carcinoma['Label'], random_state=42)
        df_ductal_carcinoma_test = df_features_test.join(df_label_test, how='inner').reset_index(drop=True)
        df_ductal_carcinoma_claib = df_features_claib.join(df_label_claib, how='inner').reset_index(drop=True)

        #lobular_carcinoma:- 
        df_features_test, df_features_claib, df_label_test, df_label_claib = train_test_split(df_lobular_carcinoma.iloc[:, :8], df_lobular_carcinoma['Label'], test_size = args.split, stratify=df_lobular_carcinoma['Label'], random_state=42)
        df_lobular_carcinoma_test = df_features_test.join(df_label_test, how='inner').reset_index(drop=True)
        df_lobular_carcinoma_claib = df_features_claib.join(df_label_claib, how='inner').reset_index(drop=True)


        #mucinous_carcinoma:-
        df_features_test, df_features_claib, df_label_test, df_label_claib = train_test_split(df_mucinous_carcinoma.iloc[:, :8], df_mucinous_carcinoma['Label'], test_size = args.split, stratify=df_mucinous_carcinoma['Label'], random_state=42)
        df_mucinous_carcinoma_test = df_features_test.join(df_label_test, how='inner').reset_index(drop=True)
        df_mucinous_carcinoma_claib = df_features_claib.join(df_label_claib, how='inner').reset_index(drop=True)


        #papillary_carcinoma:-
        df_features_test, df_features_claib, df_label_test, df_label_claib = train_test_split(df_papillary_carcinoma.iloc[:, :8], df_papillary_carcinoma['Label'], test_size = args.split, stratify=df_papillary_carcinoma['Label'], random_state=42)
        df_papillary_carcinoma_test = df_features_test.join(df_label_test, how='inner').reset_index(drop=True)
        df_papillary_carcinoma_claib = df_features_claib.join(df_label_claib, how='inner').reset_index(drop=True)



        df_test = pd.concat([df_adenosis_test, df_fibroadenoma_test, df_phyllodes_tumor_test, df_tubular_adenoma_test, df_ductal_carcinoma_test, df_lobular_carcinoma_test, df_mucinous_carcinoma_test, df_papillary_carcinoma_test], axis=0, ignore_index=True)
        df_test = df_test.reset_index(drop=True)




        prob_output = df_test.iloc[:,:-1]
        df_np = prob_output.values 
        df_prob_output_test = torch.tensor(df_np, dtype=torch.float32)

        true_class = df_test.iloc[:,-1]
        df_np = true_class.values
        df_true_class_test = torch.tensor(df_np, dtype=torch.int)







        if args.CP_method == 'THR':
            
            #adenosis_quantile_value:- 

            prob_output = df_adenosis_claib.iloc[:,:-1]
            df_np = prob_output.values 
            df_prob_output_calib = torch.tensor(df_np, dtype=torch.float32)


            true_class = df_adenosis_claib.iloc[:,-1]
            df_np = true_class.values
            df_true_class_calib = torch.tensor(df_np, dtype=torch.int)

            conformal_wrapper = THR(df_prob_output_calib, df_true_class_calib, args.alpha)
            adenosis_quantile_value = conformal_wrapper.quantile()


            #adenosis_conformal_set:-
            adenosis_conformal_set = conformal_wrapper.prediction(df_prob_output_test, adenosis_quantile_value)






            #fibroadenoma_quantile_value:- 

            prob_output = df_fibroadenoma_claib.iloc[:,:-1]
            df_np = prob_output.values 
            df_prob_output_calib = torch.tensor(df_np, dtype=torch.float32)


            true_class = df_fibroadenoma_claib.iloc[:,-1]
            df_np = true_class.values
            df_true_class_calib = torch.tensor(df_np, dtype=torch.int)

            conformal_wrapper = THR(df_prob_output_calib, df_true_class_calib, args.alpha)
            fibroadenoma_quantile_value = conformal_wrapper.quantile()


            #fibroadenoma_conformal_set:-
            fibroadenoma_conformal_set = conformal_wrapper.prediction(df_prob_output_test, fibroadenoma_quantile_value)









            #phyllodes_tumor_value:- 

            prob_output = df_phyllodes_tumor_claib.iloc[:,:-1]
            df_np = prob_output.values 
            df_prob_output_calib = torch.tensor(df_np, dtype=torch.float32)


            true_class = df_phyllodes_tumor_claib.iloc[:,-1]
            df_np = true_class.values
            df_true_class_calib = torch.tensor(df_np, dtype=torch.int)

            conformal_wrapper = THR(df_prob_output_calib, df_true_class_calib, args.alpha)
            phyllodes_tumor_value = conformal_wrapper.quantile()


            #phyllodes_conformal_set:-
            phyllodes_conformal_set = conformal_wrapper.prediction(df_prob_output_test, phyllodes_tumor_value)









            #tubular_adenoma_value:- 

            prob_output = df_tubular_adenoma_claib.iloc[:,:-1]
            df_np = prob_output.values 
            df_prob_output_calib = torch.tensor(df_np, dtype=torch.float32)


            true_class = df_tubular_adenoma_claib.iloc[:,-1]
            df_np = true_class.values
            df_true_class_calib = torch.tensor(df_np, dtype=torch.int)

            conformal_wrapper = THR(df_prob_output_calib, df_true_class_calib, args.alpha)
            tubular_adenoma_value = conformal_wrapper.quantile()


            #tubular_conformal_set:-
            tubular_conformal_set = conformal_wrapper.prediction(df_prob_output_test, tubular_adenoma_value)








            #ductal_carcinoma_value:- 

            prob_output = df_ductal_carcinoma_claib.iloc[:,:-1]
            df_np = prob_output.values 
            df_prob_output_calib = torch.tensor(df_np, dtype=torch.float32)


            true_class = df_ductal_carcinoma_claib.iloc[:,-1]
            df_np = true_class.values
            df_true_class_calib = torch.tensor(df_np, dtype=torch.int)

            conformal_wrapper = THR(df_prob_output_calib, df_true_class_calib, args.alpha)
            ductal_carcinoma_value = conformal_wrapper.quantile()

            #ductal_conformal_set:-
            ductal_conformal_set = conformal_wrapper.prediction(df_prob_output_test, ductal_carcinoma_value)










            #lobular_carcinoma_value:- 

            prob_output = df_lobular_carcinoma_claib.iloc[:,:-1]
            df_np = prob_output.values 
            df_prob_output_calib = torch.tensor(df_np, dtype=torch.float32)


            true_class = df_lobular_carcinoma_claib.iloc[:,-1]
            df_np = true_class.values
            df_true_class_calib = torch.tensor(df_np, dtype=torch.int)

            conformal_wrapper = THR(df_prob_output_calib, df_true_class_calib, args.alpha)
            lobular_carcinoma_value = conformal_wrapper.quantile()


            #lobular_carcinoma_conformal_set:-
            lobular_carcinoma_conformal_set = conformal_wrapper.prediction(df_prob_output_test, lobular_carcinoma_value)









            #mucinous_carcinoma_value:- 

            prob_output = df_mucinous_carcinoma_claib.iloc[:,:-1]
            df_np = prob_output.values 
            df_prob_output_calib = torch.tensor(df_np, dtype=torch.float32)


            true_class = df_mucinous_carcinoma_claib.iloc[:,-1]
            df_np = true_class.values
            df_true_class_calib = torch.tensor(df_np, dtype=torch.int)

            conformal_wrapper = THR(df_prob_output_calib, df_true_class_calib, args.alpha)
            mucinous_carcinoma_value = conformal_wrapper.quantile()

            #mucinous_carcinoma_conformal_set:-
            mucinous_carcinoma_conformal_set = conformal_wrapper.prediction(df_prob_output_test, mucinous_carcinoma_value)








            #papillary_carcinoma_value:- 

            prob_output = df_papillary_carcinoma_claib.iloc[:,:-1]
            df_np = prob_output.values 
            df_prob_output_calib = torch.tensor(df_np, dtype=torch.float32)


            true_class = df_papillary_carcinoma_claib.iloc[:,-1]
            df_np = true_class.values
            df_true_class_calib = torch.tensor(df_np, dtype=torch.int)

            conformal_wrapper = THR(df_prob_output_calib, df_true_class_calib, args.alpha)
            papillary_carcinoma_value = conformal_wrapper.quantile()


            #papillary_carcinoma_conformal_set:-
            papillary_carcinoma_conformal_set = conformal_wrapper.prediction(df_prob_output_test, papillary_carcinoma_value)






            #conformal_set = torch.logical_or(adenosis_conformal_set, fibroadenoma_conformal_set, phyllodes_conformal_set, tubular_conformal_set, ductal_conformal_set, lobular_carcinoma_conformal_set, mucinous_carcinoma_conformal_set, papillary_carcinoma_conformal_set).int()

            # Perform logical OR operation iteratively
            conformal_set = torch.logical_or(adenosis_conformal_set, fibroadenoma_conformal_set)
            conformal_set = torch.logical_or(conformal_set, phyllodes_conformal_set)
            conformal_set = torch.logical_or(conformal_set, tubular_conformal_set)
            conformal_set = torch.logical_or(conformal_set, ductal_conformal_set)
            conformal_set = torch.logical_or(conformal_set, lobular_carcinoma_conformal_set)
            conformal_set = torch.logical_or(conformal_set, mucinous_carcinoma_conformal_set)
            conformal_set = torch.logical_or(conformal_set, papillary_carcinoma_conformal_set)

            # Convert to integer type
            conformal_set = conformal_set.int()





        if args.expt_no == 3:
            label = df_true_class_test
            indices_0 = torch.nonzero(label == 0).squeeze()
            indices_1 = torch.nonzero(label == 1).squeeze()
            indices_2 = torch.nonzero(label == 2).squeeze()
            indices_3 = torch.nonzero(label == 3).squeeze()
            indices_4 = torch.nonzero(label == 4).squeeze()
            indices_5 = torch.nonzero(label == 5).squeeze()
            indices_6 = torch.nonzero(label == 6).squeeze()
            indices_7 = torch.nonzero(label == 7).squeeze()



            adenosis_idx = indices_0
            fibroadenoma_idx = indices_1
            phyllodes_tumor_idx = indices_2
            tubular_adenoma_idx = indices_3

            ductal_carcinoma_idx = indices_4
            lobular_carcinoma_idx = indices_5
            mucinous_carcinoma_idx = indices_6
            papillary_carcinoma_idx = indices_7



            adenosis_conformal_prediction_set = conformal_set[adenosis_idx, :]
            fibroadenoma_conformal_prediction_set = conformal_set[fibroadenoma_idx, :]
            phyllodes_tumor_conformal_prediction_set = conformal_set[phyllodes_tumor_idx, :]
            tubular_adenoma_conformal_prediction_set = conformal_set[tubular_adenoma_idx, :]

            ductal_carcinoma_conformal_prediction_set = conformal_set[ductal_carcinoma_idx, :]
            lobular_carcinoma_conformal_prediction_set = conformal_set[lobular_carcinoma_idx, :]
            mucinous_carcinoma_conformal_prediction_set = conformal_set[mucinous_carcinoma_idx, :]
            papillary_carcinoma_conformal_prediction_set = conformal_set[papillary_carcinoma_idx, :]



            adenosis_avg_set_size_len = avg_set_size_metric(adenosis_conformal_prediction_set)
            fibroadenoma_avg_set_size_len = avg_set_size_metric(fibroadenoma_conformal_prediction_set)
            phyllodes_tumor_avg_set_size_len = avg_set_size_metric(phyllodes_tumor_conformal_prediction_set)
            tubular_adenoma_avg_set_size_len = avg_set_size_metric(tubular_adenoma_conformal_prediction_set)


            ductal_carcinoma_avg_set_size_len = avg_set_size_metric(ductal_carcinoma_conformal_prediction_set)
            lobular_carcinoma_avg_set_size_len = avg_set_size_metric(lobular_carcinoma_conformal_prediction_set)
            mucinous_carcinoma_avg_set_size_len = avg_set_size_metric(mucinous_carcinoma_conformal_prediction_set)
            papillary_carcinoma_avg_set_size_len = avg_set_size_metric(papillary_carcinoma_conformal_prediction_set)


            

            adenosis_true_class = df_true_class_test[adenosis_idx]
            fibroadenoma_true_class = df_true_class_test[fibroadenoma_idx]
            phyllodes_tumor_true_class = df_true_class_test[phyllodes_tumor_idx]
            tubular_adenoma_true_class = df_true_class_test[tubular_adenoma_idx]

            ductal_carcinoma_true_class = df_true_class_test[ductal_carcinoma_idx]
            lobular_carcinoma_true_class = df_true_class_test[lobular_carcinoma_idx]
            mucinous_carcinoma_true_class = df_true_class_test[mucinous_carcinoma_idx]
            papillary_carcinoma_true_class = df_true_class_test[papillary_carcinoma_idx]



            adenosis_coverage_gap, adenosis_coverage = coverage_gap_metric(adenosis_conformal_prediction_set, adenosis_true_class, args.alpha)
            fibroadenoma_coverage_gap, fibroadenoma_coverage = coverage_gap_metric(fibroadenoma_conformal_prediction_set, fibroadenoma_true_class, args.alpha)
            phyllodes_tumor_coverage_gap, phyllodes_tumor_coverage = coverage_gap_metric(phyllodes_tumor_conformal_prediction_set, phyllodes_tumor_true_class, args.alpha)
            tubular_adenoma_coverage_gap, tubular_adenoma_coverage = coverage_gap_metric(tubular_adenoma_conformal_prediction_set, tubular_adenoma_true_class, args.alpha)

            ductal_carcinoma_coverage_gap, ductal_carcinoma_coverage = coverage_gap_metric(ductal_carcinoma_conformal_prediction_set, ductal_carcinoma_true_class, args.alpha)
            lobular_carcinoma_coverage_gap, lobular_carcinoma_coverage = coverage_gap_metric(lobular_carcinoma_conformal_prediction_set, lobular_carcinoma_true_class, args.alpha)
            mucinous_carcinoma_coverage_gap, mucinous_carcinoma_coverage = coverage_gap_metric(mucinous_carcinoma_conformal_prediction_set, mucinous_carcinoma_true_class, args.alpha)
            papillary_carcinoma_coverage_gap, papillary_carcinoma_coverage = coverage_gap_metric(papillary_carcinoma_conformal_prediction_set, papillary_carcinoma_true_class, args.alpha)

            


            



            adenosis_avg_set_size_len_for_T_trials.append(adenosis_avg_set_size_len)
            fibroadenoma_avg_set_size_len_for_T_trials.append(fibroadenoma_avg_set_size_len)
            phyllodes_tumor_avg_set_size_len_for_T_trials.append(phyllodes_tumor_avg_set_size_len)
            tubular_adenoma_avg_set_size_len_for_T_trials.append(tubular_adenoma_avg_set_size_len)

            ductal_carcinoma_avg_set_size_len_for_T_trials.append(ductal_carcinoma_avg_set_size_len)
            lobular_carcinoma_avg_set_size_len_for_T_trials.append(lobular_carcinoma_avg_set_size_len)
            mucinous_carcinoma_avg_set_size_len_for_T_trials.append(mucinous_carcinoma_avg_set_size_len)
            papillary_carcinoma_avg_set_size_len_for_T_trials.append(papillary_carcinoma_avg_set_size_len)




            adenosis_coverage_for_T_trials.append(adenosis_coverage)
            fibroadenoma_coverage_for_T_trials.append(fibroadenoma_coverage)
            phyllodes_tumor_coverage_for_T_trials.append(phyllodes_tumor_coverage)
            tubular_adenoma_coverage_for_T_trials.append(tubular_adenoma_coverage)

            ductal_carcinoma_coverage_for_T_trials.append(ductal_carcinoma_coverage)
            lobular_carcinoma_coverage_for_T_trials.append(lobular_carcinoma_coverage)
            mucinous_carcinoma_coverage_for_T_trials.append(mucinous_carcinoma_coverage)
            papillary_carcinoma_coverage_for_T_trials.append(papillary_carcinoma_coverage)




    if args.expt_no == 3:

        print()
        print()
        print(f'coverage:-')

        adenosis_coverage_for_T_trials = np.array(adenosis_coverage_for_T_trials)
        adenosis_average_coverage = np.mean(adenosis_coverage_for_T_trials)
        adenosis_std_dev_coverage = np.std(adenosis_coverage_for_T_trials, ddof=1)

        print()
        print(f"adenosis_average_coverage: {adenosis_average_coverage}")
        print(f"adenosis_std_dev_coverage: {adenosis_std_dev_coverage}")


        fibroadenoma_coverage_for_T_trials = np.array(fibroadenoma_coverage_for_T_trials)
        fibroadenoma_average_coverage = np.mean(fibroadenoma_coverage_for_T_trials)
        fibroadenoma_std_dev_coverage = np.std(fibroadenoma_coverage_for_T_trials, ddof=1)

        print()
        print(f"fibroadenoma_average_coverage: {fibroadenoma_average_coverage}")
        print(f"fibroadenoma_std_dev_coverage: {fibroadenoma_std_dev_coverage}")
    


        phyllodes_tumor_coverage_for_T_trials = np.array(phyllodes_tumor_coverage_for_T_trials)
        phyllodes_tumor_average_coverage = np.mean(phyllodes_tumor_coverage_for_T_trials)
        phyllodes_tumor_std_dev_coverage = np.std(phyllodes_tumor_coverage_for_T_trials, ddof=1)

        print()
        print(f"phyllodes_tumor_average_coverage: {phyllodes_tumor_average_coverage}")
        print(f"phyllodes_tumor_std_dev_coverage: {phyllodes_tumor_std_dev_coverage}")
    


        tubular_adenoma_coverage_for_T_trials = np.array(tubular_adenoma_coverage_for_T_trials)
        tubular_adenoma_average_coverage = np.mean(tubular_adenoma_coverage_for_T_trials)
        tubular_adenoma_std_dev_coverage = np.std(tubular_adenoma_coverage_for_T_trials, ddof=1)

        print()
        print(f"tubular_adenoma_average_coverage: {tubular_adenoma_average_coverage}")
        print(f"tubular_adenoma_std_dev_coverage: {tubular_adenoma_std_dev_coverage}")


        ductal_carcinoma_coverage_for_T_trials = np.array(ductal_carcinoma_coverage_for_T_trials)
        ductal_carcinoma_average_coverage = np.mean(ductal_carcinoma_coverage_for_T_trials)
        ductal_carcinoma_std_dev_coverage = np.std(ductal_carcinoma_coverage_for_T_trials, ddof=1)

        print()
        print(f"ductal_carcinoma_average_coverage: {ductal_carcinoma_average_coverage}")
        print(f"ductal_carcinoma_std_dev_coverage: {ductal_carcinoma_std_dev_coverage}")



        lobular_carcinoma_coverage_for_T_trials = np.array(lobular_carcinoma_coverage_for_T_trials)
        lobular_carcinoma_average_coverage = np.mean(lobular_carcinoma_coverage_for_T_trials)
        lobular_carcinoma_std_dev_coverage = np.std(lobular_carcinoma_coverage_for_T_trials, ddof=1)

        print()
        print(f"lobular_carcinoma_average_coverage: {lobular_carcinoma_average_coverage}")
        print(f"lobular_carcinoma_std_dev_coverage: {lobular_carcinoma_std_dev_coverage}")


        mucinous_carcinoma_coverage_for_T_trials = np.array(mucinous_carcinoma_coverage_for_T_trials)
        mucinous_carcinoma_average_coverage = np.mean(mucinous_carcinoma_coverage_for_T_trials)
        mucinous_carcinoma_std_dev_coverage = np.std(mucinous_carcinoma_coverage_for_T_trials, ddof=1)

        print()
        print(f"mucinous_carcinoma_average_coverage: {mucinous_carcinoma_average_coverage}")
        print(f"mucinous_carcinoma_std_dev_coverage: {mucinous_carcinoma_std_dev_coverage}")



        papillary_carcinoma_coverage_for_T_trials = np.array(papillary_carcinoma_coverage_for_T_trials)
        papillary_carcinoma_average_coverage = np.mean(papillary_carcinoma_coverage_for_T_trials)
        papillary_carcinoma_std_dev_coverage = np.std(papillary_carcinoma_coverage_for_T_trials, ddof=1)

        print()
        print(f"papillary_carcinoma_average_coverage: {papillary_carcinoma_average_coverage}")
        print(f"papillary_carcinoma_std_dev_coverage: {papillary_carcinoma_std_dev_coverage}")








        print()
        print()
        print(f'set_size :-')
    
        adenosis_avg_set_size_len_for_T_trials= np.array(adenosis_avg_set_size_len_for_T_trials)
        adenosis_average_set_size_len = np.mean(adenosis_avg_set_size_len_for_T_trials)
        adenosis_std_dev_set_size_len = np.std(adenosis_avg_set_size_len_for_T_trials, ddof=1)
    
        print()
        print(f"adenosis_average_set_size_len: {adenosis_average_set_size_len}")
        print(f"adenosis_std_dev_set_size_len: {adenosis_std_dev_set_size_len}")
    


        fibroadenoma_avg_set_size_len_for_T_trials = np.array(fibroadenoma_avg_set_size_len_for_T_trials)
        fibroadenoma_average_set_size_len = np.mean(fibroadenoma_avg_set_size_len_for_T_trials)
        fibroadenoma_std_dev_set_size_len = np.std(fibroadenoma_avg_set_size_len_for_T_trials, ddof=1)
    
        print()
        print(f"fibroadenoma_average_set_size_len: {fibroadenoma_average_set_size_len}")
        print(f"fibroadenoma_std_dev_set_size_len: {fibroadenoma_std_dev_set_size_len}")
    


        phyllodes_tumor_avg_set_size_len_for_T_trials = np.array(phyllodes_tumor_avg_set_size_len_for_T_trials)
        phyllodes_average_set_size_len = np.mean(phyllodes_tumor_avg_set_size_len_for_T_trials)
        phyllodes_std_dev_set_size_len = np.std(phyllodes_tumor_avg_set_size_len_for_T_trials, ddof=1)
    
        print()
        print(f"phyllodes_average_set_size_len: {phyllodes_average_set_size_len}")
        print(f"phyllodes_std_dev_set_size_len: {phyllodes_std_dev_set_size_len}")
    


        tubular_adenoma_avg_set_size_len_for_T_trials = np.array(tubular_adenoma_avg_set_size_len_for_T_trials)
        tubular_adenoma_average_set_size_len = np.mean(tubular_adenoma_avg_set_size_len_for_T_trials)
        tubular_adenoma_std_dev_set_size_len = np.std(tubular_adenoma_avg_set_size_len_for_T_trials, ddof=1)
    
        print()
        print(f"tubular_adenoma_average_set_size_len: {tubular_adenoma_average_set_size_len}")
        print(f"tubular_adenoma_std_dev_set_size_len: {tubular_adenoma_std_dev_set_size_len}")



        ductal_carcinoma_avg_set_size_len_for_T_trials = np.array(ductal_carcinoma_avg_set_size_len_for_T_trials)
        ductal_carcinoma_average_set_size_len = np.mean(ductal_carcinoma_avg_set_size_len_for_T_trials)
        ductal_carcinoma_std_dev_set_size_len = np.std(ductal_carcinoma_avg_set_size_len_for_T_trials, ddof=1)
    
        print()
        print(f"ductal_carcinoma_average_set_size_len: {ductal_carcinoma_average_set_size_len}")
        print(f"ductal_carcinoma_std_dev_set_size_len: {ductal_carcinoma_std_dev_set_size_len}")
    


        lobular_carcinoma_avg_set_size_len_for_T_trials = np.array(lobular_carcinoma_avg_set_size_len_for_T_trials)
        lobular_carcinoma_average_set_size_len = np.mean(lobular_carcinoma_avg_set_size_len_for_T_trials)
        lobular_carcinoma_std_dev_set_size_len = np.std(lobular_carcinoma_avg_set_size_len_for_T_trials, ddof=1)
    
        print()
        print(f"lobular_carcinoma_average_set_size_len: {lobular_carcinoma_average_set_size_len}")
        print(f"lobular_carcinoma_std_dev_set_size_len: {lobular_carcinoma_std_dev_set_size_len}")



        mucinous_carcinoma_avg_set_size_len_for_T_trials = np.array(mucinous_carcinoma_avg_set_size_len_for_T_trials)
        mucinous_carcinoma_average_set_size_len = np.mean(mucinous_carcinoma_avg_set_size_len_for_T_trials)
        mucinous_carcinoma_std_dev_set_size_len = np.std(mucinous_carcinoma_avg_set_size_len_for_T_trials, ddof=1)
    
        print()
        print(f"mucinous_carcinoma_average_set_size_len: {mucinous_carcinoma_average_set_size_len}")
        print(f"mucinous_carcinoma_std_dev_set_size_len: {mucinous_carcinoma_std_dev_set_size_len}")
    


        papillary_carcinoma_avg_set_size_len_for_T_trials = np.array(papillary_carcinoma_avg_set_size_len_for_T_trials)
        papillary_carcinoma_average_set_size_len = np.mean(papillary_carcinoma_avg_set_size_len_for_T_trials)
        papillary_carcinoma_std_dev_set_size_len = np.std(papillary_carcinoma_avg_set_size_len_for_T_trials, ddof=1)
    
        print()
        print(f"papillary_carcinoma_average_set_size_len: {papillary_carcinoma_average_set_size_len}")
        print(f"papillary_carcinoma_std_dev_set_size_len: {papillary_carcinoma_std_dev_set_size_len}")

        







if __name__ == '__main__':
    main()











