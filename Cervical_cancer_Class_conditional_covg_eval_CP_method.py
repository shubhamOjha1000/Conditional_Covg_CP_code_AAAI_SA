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

        df_Superficial = df[df['Label'] == 0]
        df_Intermediate = df[df['Label'] == 1]
        df_Parabasal = df[df['Label'] == 2]
        df_ASC_US = df[df['Label'] == 3]
        df_ASC_H = df[df['Label'] == 4]
        df_LSIL = df[df['Label'] == 5]
        df_HSIL = df[df['Label'] == 6]
        df_SCC = df[df['Label'] == 7]


        #Superficial:- 
        df_features_test, df_features_claib, df_label_test, df_label_claib = train_test_split(df_Superficial.iloc[:, :8], df_Superficial['Label'], test_size = args.split, stratify=df_Superficial['Label'], random_state=42)
        df_Superficial_test = df_features_test.join(df_label_test, how='inner').reset_index(drop=True)
        df_Superficial_claib = df_features_claib.join(df_label_claib, how='inner').reset_index(drop=True)


        #Intermediate:-
        df_features_test, df_features_claib, df_label_test, df_label_claib = train_test_split(df_Intermediate.iloc[:, :8], df_Intermediate['Label'], test_size = args.split, stratify=df_Intermediate['Label'], random_state=42)
        df_Intermediate_test = df_features_test.join(df_label_test, how='inner').reset_index(drop=True)
        df_Intermediate_claib = df_features_claib.join(df_label_claib, how='inner').reset_index(drop=True)


        #Parabasal:-
        df_features_test, df_features_claib, df_label_test, df_label_claib = train_test_split(df_Parabasal.iloc[:, :8], df_Parabasal['Label'], test_size = args.split, stratify=df_Parabasal['Label'], random_state=42)
        df_Parabasal_test = df_features_test.join(df_label_test, how='inner').reset_index(drop=True)
        df_Parabasal_claib = df_features_claib.join(df_label_claib, how='inner').reset_index(drop=True)


        #ASC_US:- 
        df_features_test, df_features_claib, df_label_test, df_label_claib = train_test_split(df_ASC_US.iloc[:, :8], df_ASC_US['Label'], test_size = args.split, stratify=df_ASC_US['Label'], random_state=42)
        df_ASC_US_test = df_features_test.join(df_label_test, how='inner').reset_index(drop=True)
        df_ASC_US_claib = df_features_claib.join(df_label_claib, how='inner').reset_index(drop=True)


        #ASC_H:-
        df_features_test, df_features_claib, df_label_test, df_label_claib = train_test_split(df_ASC_H.iloc[:, :8], df_ASC_H['Label'], test_size = args.split, stratify=df_ASC_H['Label'], random_state=42)
        df_ASC_H_test = df_features_test.join(df_label_test, how='inner').reset_index(drop=True)
        df_ASC_H_claib = df_features_claib.join(df_label_claib, how='inner').reset_index(drop=True)

        #LSIL:-
        df_features_test, df_features_claib, df_label_test, df_label_claib = train_test_split(df_LSIL.iloc[:, :8], df_LSIL['Label'], test_size = args.split, stratify=df_LSIL['Label'], random_state=42)
        df_LSIL_test = df_features_test.join(df_label_test, how='inner').reset_index(drop=True)
        df_LSIL_claib = df_features_claib.join(df_label_claib, how='inner').reset_index(drop=True)


        #HSIL:-
        df_features_test, df_features_claib, df_label_test, df_label_claib = train_test_split(df_HSIL.iloc[:, :8], df_HSIL['Label'], test_size = args.split, stratify=df_HSIL['Label'], random_state=42)
        df_HSIL_test = df_features_test.join(df_label_test, how='inner').reset_index(drop=True)
        df_HSIL_claib = df_features_claib.join(df_label_claib, how='inner').reset_index(drop=True)


        #SCC:- 
        df_features_test, df_features_claib, df_label_test, df_label_claib = train_test_split(df_SCC.iloc[:, :8], df_SCC['Label'], test_size = args.split, stratify=df_SCC['Label'], random_state=42)
        df_SCC_test = df_features_test.join(df_label_test, how='inner').reset_index(drop=True)
        df_SCC_claib = df_features_claib.join(df_label_claib, how='inner').reset_index(drop=True)





        df_test = pd.concat([df_Superficial_test, df_Intermediate_test, df_Parabasal_test, df_ASC_US_test, df_ASC_H_test, df_LSIL_test, df_HSIL_test, df_SCC_test], axis=0, ignore_index=True)
        df_test = df_test.reset_index(drop=True)




        prob_output = df_test.iloc[:,:-1]
        df_np = prob_output.values 
        df_prob_output_test = torch.tensor(df_np, dtype=torch.float32)

        true_class = df_test.iloc[:,-1]
        df_np = true_class.values
        df_true_class_test = torch.tensor(df_np, dtype=torch.int)





        if args.CP_method == 'THR':


            #Superficial_quantile_value:- 
            prob_output = df_Superficial_claib.iloc[:,:-1]
            df_np = prob_output.values 
            df_prob_output_calib = torch.tensor(df_np, dtype=torch.float32)

            true_class = df_Superficial_claib.iloc[:,-1]
            df_np = true_class.values
            df_true_class_calib = torch.tensor(df_np, dtype=torch.int)

            conformal_wrapper = THR(df_prob_output_calib, df_true_class_calib, args.alpha)
            Superficial_quantile_value = conformal_wrapper.quantile()

            #Superficial_conformal_set:-
            Superficial_conformal_set = conformal_wrapper.prediction(df_prob_output_test, Superficial_quantile_value)





            #Intermediate_quantile_value:- 
            prob_output = df_Intermediate_claib.iloc[:,:-1]
            df_np = prob_output.values 
            df_prob_output_calib = torch.tensor(df_np, dtype=torch.float32)

            true_class = df_Intermediate_claib.iloc[:,-1]
            df_np = true_class.values
            df_true_class_calib = torch.tensor(df_np, dtype=torch.int)

            conformal_wrapper = THR(df_prob_output_calib, df_true_class_calib, args.alpha)
            Intermediate_quantile_value = conformal_wrapper.quantile()

            #Intermediate_conformal_set:-
            Intermediate_conformal_set = conformal_wrapper.prediction(df_prob_output_test, Intermediate_quantile_value)





            #Parabasal_quantile_value:- 
            prob_output = df_Parabasal_claib.iloc[:,:-1]
            df_np = prob_output.values 
            df_prob_output_calib = torch.tensor(df_np, dtype=torch.float32)

            true_class = df_Parabasal_claib.iloc[:,-1]
            df_np = true_class.values
            df_true_class_calib = torch.tensor(df_np, dtype=torch.int)

            conformal_wrapper = THR(df_prob_output_calib, df_true_class_calib, args.alpha)
            Parabasal_quantile_value = conformal_wrapper.quantile()

            #Parabasal_conformal_set:-
            Parabasal_conformal_set = conformal_wrapper.prediction(df_prob_output_test, Parabasal_quantile_value)





            #ASC_US_quantile_value:- 
            prob_output = df_ASC_US_claib.iloc[:,:-1]
            df_np = prob_output.values 
            df_prob_output_calib = torch.tensor(df_np, dtype=torch.float32)

            true_class = df_ASC_US_claib.iloc[:,-1]
            df_np = true_class.values
            df_true_class_calib = torch.tensor(df_np, dtype=torch.int)

            conformal_wrapper = THR(df_prob_output_calib, df_true_class_calib, args.alpha)
            ASC_US_quantile_value = conformal_wrapper.quantile()

            #ASC_US_conformal_set:-
            ASC_US_conformal_set = conformal_wrapper.prediction(df_prob_output_test, ASC_US_quantile_value)




            #ASC_H_quantile_value:- 
            prob_output = df_ASC_H_claib.iloc[:,:-1]
            df_np = prob_output.values 
            df_prob_output_calib = torch.tensor(df_np, dtype=torch.float32)

            true_class = df_ASC_H_claib.iloc[:,-1]
            df_np = true_class.values
            df_true_class_calib = torch.tensor(df_np, dtype=torch.int)

            conformal_wrapper = THR(df_prob_output_calib, df_true_class_calib, args.alpha)
            ASC_H_quantile_value = conformal_wrapper.quantile()

            #ASC_H_conformal_set:-
            ASC_H_conformal_set = conformal_wrapper.prediction(df_prob_output_test, ASC_H_quantile_value)





            #LSIL_quantile_value:- 
            prob_output = df_LSIL_claib.iloc[:,:-1]
            df_np = prob_output.values 
            df_prob_output_calib = torch.tensor(df_np, dtype=torch.float32)

            true_class = df_LSIL_claib.iloc[:,-1]
            df_np = true_class.values
            df_true_class_calib = torch.tensor(df_np, dtype=torch.int)

            conformal_wrapper = THR(df_prob_output_calib, df_true_class_calib, args.alpha)
            LSIL_quantile_value = conformal_wrapper.quantile()

            #LSIL_conformal_set:-
            LSIL_quantile_value_conformal_set = conformal_wrapper.prediction(df_prob_output_test, LSIL_quantile_value)





            #HSIL_quantile_value:- 
            prob_output = df_HSIL_claib.iloc[:,:-1]
            df_np = prob_output.values 
            df_prob_output_calib = torch.tensor(df_np, dtype=torch.float32)

            true_class = df_HSIL_claib.iloc[:,-1]
            df_np = true_class.values
            df_true_class_calib = torch.tensor(df_np, dtype=torch.int)

            conformal_wrapper = THR(df_prob_output_calib, df_true_class_calib, args.alpha)
            HSIL_quantile_value = conformal_wrapper.quantile()

            #HSIL_conformal_set:-
            HSIL_quantile_value_conformal_set = conformal_wrapper.prediction(df_prob_output_test, HSIL_quantile_value)





            #SCC_quantile_value:- 
            prob_output = df_SCC_claib.iloc[:,:-1]
            df_np = prob_output.values 
            df_prob_output_calib = torch.tensor(df_np, dtype=torch.float32)

            true_class = df_SCC_claib.iloc[:,-1]
            df_np = true_class.values
            df_true_class_calib = torch.tensor(df_np, dtype=torch.int)

            conformal_wrapper = THR(df_prob_output_calib, df_true_class_calib, args.alpha)
            SCC_quantile_value = conformal_wrapper.quantile()

            #SCC_conformal_set:-
            SCC_quantile_value_conformal_set = conformal_wrapper.prediction(df_prob_output_test, SCC_quantile_value)





            # Perform logical OR operation iteratively

            conformal_set = torch.logical_or(Superficial_conformal_set, Intermediate_conformal_set)
            conformal_set = torch.logical_or(conformal_set, Parabasal_conformal_set)
            conformal_set = torch.logical_or(conformal_set, ASC_US_conformal_set)
            conformal_set = torch.logical_or(conformal_set, ASC_H_conformal_set)
            conformal_set = torch.logical_or(conformal_set, LSIL_quantile_value_conformal_set)
            conformal_set = torch.logical_or(conformal_set, HSIL_quantile_value_conformal_set)
            conformal_set = torch.logical_or(conformal_set, SCC_quantile_value_conformal_set)


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








    if args.expt_no == 3:

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