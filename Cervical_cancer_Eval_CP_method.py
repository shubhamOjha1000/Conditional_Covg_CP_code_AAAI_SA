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


    avg_set_size_len_for_T_trials = []
    avg_coverage_gap_for_T_trials = []
    avg_coverage_for_T_trials = []



    Superficial_coverage_for_T_trials = []
    Intermediate_coverage_for_T_trials = []
    Parabasal_coverage_for_T_trials = []
    ASC_US_coverage_for_T_trials = []
    ASC_H_coverage_for_T_trials = []
    LSIL_coverage_for_T_trials = []
    HSIL_coverage_for_T_trials = []
    SCC_coverage_for_T_trials = []



    Superficial_avg_set_size_len_for_T_trials = []
    Intermediate_avg_set_size_len_for_T_trials = []
    Parabasal_avg_set_size_len_for_T_trials = []
    ASC_US_avg_set_size_len_for_T_trials = []
    ASC_H_avg_set_size_len_for_T_trials = []
    LSIL_avg_set_size_len_for_T_trials = []
    HSIL_avg_set_size_len_for_T_trials = []
    SCC_avg_set_size_len_for_T_trials = []
    



    for t in range(args.Trials):
        print()
        print(f'Trials :- {t}')
        print()

        # loading the annotation file :-
        df = pd.read_csv(args.softmax_output_file_path)
        df = df.sample(frac=1).reset_index(drop=True)


        # calib-test split :- 
        feature_test, feature_calib = train_test_split(df, test_size = args.split, stratify=df['Label'], random_state=42)

        feature_test = feature_test.reset_index(drop=True)
        feature_calib = feature_calib.reset_index(drop=True)

        prob_output = feature_calib.iloc[:,:-1]
        df_np = prob_output.values 
        df_prob_output_calib = torch.tensor(df_np, dtype=torch.float32)

        prob_output = feature_test.iloc[:,:-1]
        df_np = prob_output.values
        df_prob_output_test = torch.tensor(df_np, dtype=torch.float32)


        true_class = feature_calib.iloc[:,-1]
        df_np = true_class.values
        df_true_class_calib = torch.tensor(df_np, dtype=torch.int)


        true_class = feature_test.iloc[:,-1]
        df_np = true_class.values
        df_true_class_test = torch.tensor(df_np, dtype=torch.int)





        if args.CP_method == 'THR':
            conformal_wrapper = THR(df_prob_output_calib, df_true_class_calib, args.alpha)
            quantile_value = conformal_wrapper.quantile()

            conformal_set = conformal_wrapper.prediction(df_prob_output_test, quantile_value)
           


            
        elif args.CP_method == 'APS':
            conformal_wrapper = APS(df_prob_output_calib, df_true_class_calib, args.alpha)
            quantile_value = conformal_wrapper.quantile()

            conformal_set = conformal_wrapper.prediction(df_prob_output_test, quantile_value)


        elif args.CP_method == 'RAPS':

            conformal_wrapper = RAPS(df_prob_output_calib, df_true_class_calib, args.alpha, args.k_reg, args.lambd, args.rand)
            quantile_value = conformal_wrapper.quantile()

            conformal_set = conformal_wrapper.prediction(df_prob_output_test, quantile_value)



        
        if args.expt_no == 1:

            avg_set_size = avg_set_size_metric(conformal_set)
            print(f'avg_set_size:- {avg_set_size}')
            
            coverage_gap, coverage = coverage_gap_metric(conformal_set, df_true_class_test, args.alpha)
            #print(f'coverage_gap:- {coverage_gap}')
            #print(f'coverage:- {coverage}')

            
            avg_set_size_len_for_T_trials.append(avg_set_size)
            avg_coverage_gap_for_T_trials.append(coverage_gap)
            avg_coverage_for_T_trials.append(coverage)

        
        elif args.expt_no == 3:
            label = df_true_class_test
            indices_0 = torch.nonzero(label == 0).squeeze()
            indices_1 = torch.nonzero(label == 1).squeeze()
            indices_2 = torch.nonzero(label == 2).squeeze()
            indices_3 = torch.nonzero(label == 3).squeeze()
            indices_4 = torch.nonzero(label == 4).squeeze()
            indices_5 = torch.nonzero(label == 5).squeeze()
            indices_6 = torch.nonzero(label == 6).squeeze()
            indices_7 = torch.nonzero(label == 7).squeeze()

            
            Superficial_idx = indices_0
            Intermediate_idx = indices_1
            Parabasal_idx = indices_2
            ASC_US_idx = indices_3
            ASC_H_idx = indices_4
            LSIL_idx = indices_5
            HSIL_idx = indices_6
            SCC_idx = indices_7

            



            Superficial_true_class = df_true_class_test[Superficial_idx]
            Intermediate_true_class = df_true_class_test[Intermediate_idx]
            Parabasal_true_class = df_true_class_test[Parabasal_idx]
            ASC_US_true_class = df_true_class_test[ASC_US_idx]
            ASC_H_true_class = df_true_class_test[ASC_H_idx]
            LSIL_true_class = df_true_class_test[LSIL_idx]
            HSIL_true_class = df_true_class_test[HSIL_idx]
            SCC_true_class = df_true_class_test[SCC_idx]




            Superficial_conformal_prediction_set = conformal_set[Superficial_idx, :]
            Intermediate_conformal_prediction_set = conformal_set[Intermediate_idx, :]
            Parabasal_conformal_prediction_set = conformal_set[Parabasal_idx, :]
            ASC_US_conformal_prediction_set = conformal_set[ASC_US_idx, :]
            ASC_H_conformal_prediction_set = conformal_set[ASC_H_idx, :]
            LSIL_conformal_prediction_set = conformal_set[LSIL_idx, :]
            HSIL_conformal_prediction_set = conformal_set[HSIL_idx, :]
            SCC_conformal_prediction_set = conformal_set[SCC_idx, :]


            Superficial_avg_set_size_len = avg_set_size_metric(Superficial_conformal_prediction_set)
            Intermediate_avg_set_size_len = avg_set_size_metric(Intermediate_conformal_prediction_set)
            Parabasal_avg_set_size_len = avg_set_size_metric(Parabasal_conformal_prediction_set)
            ASC_US_avg_set_size_len = avg_set_size_metric(ASC_US_conformal_prediction_set)
            ASC_H_avg_set_size_len = avg_set_size_metric(ASC_H_conformal_prediction_set)
            LSIL_avg_set_size_len = avg_set_size_metric(LSIL_conformal_prediction_set)
            HSIL_avg_set_size_len = avg_set_size_metric(HSIL_conformal_prediction_set)
            SCC_avg_set_size_len = avg_set_size_metric(SCC_conformal_prediction_set)



            _, Superficial_coverage = coverage_gap_metric(Superficial_conformal_prediction_set, Superficial_true_class, args.alpha)
            _, Intermediate_coverage = coverage_gap_metric(Intermediate_conformal_prediction_set, Intermediate_true_class, args.alpha)
            _, Parabasal_coverage = coverage_gap_metric(Parabasal_conformal_prediction_set, Parabasal_true_class, args.alpha)
            _, ASC_US_coverage = coverage_gap_metric(ASC_US_conformal_prediction_set, ASC_US_true_class, args.alpha)
            _, ASC_H_coverage = coverage_gap_metric(ASC_H_conformal_prediction_set, ASC_H_true_class, args.alpha)
            _, LSIL_coverage = coverage_gap_metric(LSIL_conformal_prediction_set, LSIL_true_class, args.alpha)
            _, HSIL_coverage = coverage_gap_metric(HSIL_conformal_prediction_set, HSIL_true_class, args.alpha)
            _, SCC_coverage = coverage_gap_metric(SCC_conformal_prediction_set, SCC_true_class, args.alpha)

            

            Superficial_avg_set_size_len_for_T_trials.append(Superficial_avg_set_size_len)
            Intermediate_avg_set_size_len_for_T_trials.append(Intermediate_avg_set_size_len)
            Parabasal_avg_set_size_len_for_T_trials.append(Parabasal_avg_set_size_len)
            ASC_US_avg_set_size_len_for_T_trials.append(ASC_US_avg_set_size_len)
            ASC_H_avg_set_size_len_for_T_trials.append(ASC_H_avg_set_size_len)
            LSIL_avg_set_size_len_for_T_trials.append(LSIL_avg_set_size_len)
            HSIL_avg_set_size_len_for_T_trials.append(HSIL_avg_set_size_len)
            SCC_avg_set_size_len_for_T_trials.append(SCC_avg_set_size_len)


            Superficial_coverage_for_T_trials.append(Superficial_coverage)
            Intermediate_coverage_for_T_trials.append(Intermediate_coverage)
            Parabasal_coverage_for_T_trials.append(Parabasal_coverage)
            ASC_US_coverage_for_T_trials.append(ASC_US_coverage)
            ASC_H_coverage_for_T_trials.append(ASC_H_coverage)
            LSIL_coverage_for_T_trials.append(LSIL_coverage)
            HSIL_coverage_for_T_trials.append(HSIL_coverage)
            SCC_coverage_for_T_trials.append(SCC_coverage)





    if args.expt_no == 1:
        avg_set_size_len_for_T_trials = np.array(avg_set_size_len_for_T_trials)
        average = np.mean(avg_set_size_len_for_T_trials)
        std_dev = np.std(avg_set_size_len_for_T_trials, ddof=1)

        print()
        print()
        print()
        print()
        print(f"Average set_size_len_for_T_trials: {average}")
        print(f"Standard Deviation set_size_len_for_T_trials: {std_dev}")




        avg_coverage_gap_for_T_trials = np.array(avg_coverage_gap_for_T_trials)
        average = np.mean(avg_coverage_gap_for_T_trials)
        std_dev = np.std(avg_coverage_gap_for_T_trials, ddof=1)

        print()
        print(f"Average coverage_gap_for_T_trials: {average}")
        print(f"Standard Deviation coverage_gap_for_T_trials: {std_dev}")




        avg_coverage_for_T_trials = np.array(avg_coverage_for_T_trials)
        average = np.mean(avg_coverage_for_T_trials)
        std_dev = np.std(avg_coverage_for_T_trials, ddof=1)

        print()
        print(f"Average coverage_for_T_trials: {average}")
        print(f"Standard Deviation coverage_for_T_trials: {std_dev}")


    
    elif args.expt_no == 3:


        print()
        print()
        print(f'coverage:-')

        Superficial_coverage_for_T_trials = np.array(Superficial_coverage_for_T_trials)
        Superficial_average_coverage = np.mean(Superficial_coverage_for_T_trials)
        Superficial_std_dev_coverage = np.std(Superficial_coverage_for_T_trials, ddof=1)
        print()
        print(f"Superficial_average_coverage: {Superficial_average_coverage}")
        print(f"Superficial_std_dev_coverage: {Superficial_std_dev_coverage}")


        Intermediate_coverage_for_T_trials = np.array(Intermediate_coverage_for_T_trials)
        Intermediate_average_coverage = np.mean(Intermediate_coverage_for_T_trials)
        Intermediate_std_dev_coverage = np.std(Intermediate_coverage_for_T_trials, ddof=1)
        print()
        print(f"Intermediate_average_coverage: {Intermediate_average_coverage}")
        print(f"Intermediate_std_dev_coverage: {Intermediate_std_dev_coverage}")


        Parabasal_coverage_for_T_trials = np.array(Parabasal_coverage_for_T_trials)
        Parabasal_average_coverage = np.mean(Parabasal_coverage_for_T_trials)
        Parabasal_std_dev_coverage = np.std(Parabasal_coverage_for_T_trials, ddof=1)
        print()
        print(f"Parabasal_average_coverage: {Parabasal_average_coverage}")
        print(f"Parabasal_std_dev_coverage: {Parabasal_std_dev_coverage}")


        ASC_US_coverage_for_T_trials = np.array(ASC_US_coverage_for_T_trials)
        ASC_US_average_coverage = np.mean(ASC_US_coverage_for_T_trials)
        ASC_US_std_dev_coverage = np.std(ASC_US_coverage_for_T_trials, ddof=1)
        print()
        print(f"ASC_US_average_coverage: {ASC_US_average_coverage}")
        print(f"ASC_US_std_dev_coverage: {ASC_US_std_dev_coverage}")


        ASC_H_coverage_for_T_trials = np.array(ASC_H_coverage_for_T_trials)
        ASC_H_average_coverage = np.mean(ASC_H_coverage_for_T_trials)
        ASC_H_std_dev_coverage = np.std(ASC_H_coverage_for_T_trials, ddof=1)
        print()
        print(f"ASC_H_average_coverage: {ASC_H_average_coverage}")
        print(f"ASC_H_std_dev_coverage: {ASC_H_std_dev_coverage}")



        LSIL_coverage_for_T_trials = np.array(LSIL_coverage_for_T_trials)
        LSIL_average_coverage = np.mean(LSIL_coverage_for_T_trials)
        LSIL_std_dev_coverage = np.std(LSIL_coverage_for_T_trials, ddof=1)
        print()
        print(f"LSIL_average_coverage: {LSIL_average_coverage}")
        print(f"LSIL_std_dev_coverage: {LSIL_std_dev_coverage}")



        HSIL_coverage_for_T_trials = np.array(HSIL_coverage_for_T_trials)
        HSIL_average_coverage = np.mean(HSIL_coverage_for_T_trials)
        HSIL_std_dev_coverage = np.std(HSIL_coverage_for_T_trials, ddof=1)
        print()
        print(f"HSIL_average_coverage: {HSIL_average_coverage}")
        print(f"HSIL_std_dev_coverage: {HSIL_std_dev_coverage}")




        SCC_coverage_for_T_trials = np.array(SCC_coverage_for_T_trials)
        SCC_average_coverage = np.mean(SCC_coverage_for_T_trials)
        SCC_std_dev_coverage = np.std(SCC_coverage_for_T_trials, ddof=1)
        print()
        print(f"SCC_average_coverage: {SCC_average_coverage}")
        print(f"SCC_std_dev_coverage: {SCC_std_dev_coverage}")



if __name__ == '__main__':
    main()