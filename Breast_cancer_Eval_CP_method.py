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


    benign_avg_set_size_len_for_T_trials = []
    benign_avg_coverage_gap_for_T_trials = []
    benign_avg_coverage_for_T_trials = []


    malignant_avg_set_size_len_for_T_trials = []
    malignant_avg_coverage_gap_for_T_trials = []
    malignant_avg_coverage_for_T_trials = []




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

        # calib-test split :- 
        test_0, calib_0, test_1, calib_1, test_2, calib_2, test_3, calib_3, test_4, calib_4, test_5, calib_5, test_6, calib_6, test_7, calib_7, test_label, calib_label = train_test_split(df['0'],df['1'], df['2'],df['3'], df['4'],df['5'], df['6'],df['7'], df['Label'], test_size = args.split, stratify=df['Label'], random_state=42)

        test_0 = test_0.to_frame()
        test_1 = test_1.to_frame()
        test_2 = test_2.to_frame()
        test_3 = test_3.to_frame()
        test_4 = test_4.to_frame()
        test_5 = test_5.to_frame()
        test_6 = test_6.to_frame()
        test_7 = test_7.to_frame()
        test = test_0.join(test_1, how='inner').join(test_2, how='inner').join(test_3, how='inner').join(test_4, how='inner').join(test_5, how='inner').join(test_6, how='inner').join(test_7, how='inner').join(test_label, how='inner')
        test = test.reset_index(drop=True)

        calib_0 = calib_0.to_frame()
        calib_1 = calib_1.to_frame()
        calib_2 = calib_2.to_frame()
        calib_3 = calib_3.to_frame()
        calib_4 = calib_4.to_frame()
        calib_5 = calib_5.to_frame()
        calib_6 = calib_6.to_frame()
        calib_7 = calib_7.to_frame()
        calib = calib_0.join(calib_1, how='inner').join(calib_2, how='inner').join(calib_3, how='inner').join(calib_4, how='inner').join(calib_5, how='inner').join(calib_6, how='inner').join(calib_7, how='inner').join(calib_label, how='inner')
        calib = calib.reset_index(drop=True)

        prob_output = calib.iloc[:,:-1]
        df_np = prob_output.values 
        df_prob_output_calib = torch.tensor(df_np, dtype=torch.float32)
        #print(f'df_prob_output_calib :- {len(df_prob_output_calib)}')


        prob_output = test.iloc[:,:-1]
        df_np = prob_output.values
        df_prob_output_test = torch.tensor(df_np, dtype=torch.float32)
        #print(f'df_prob_output_test :- {len(df_prob_output_test)}')



        true_class = calib.iloc[:,-1]
        df_np = true_class.values
        df_true_class_calib = torch.tensor(df_np, dtype=torch.int)
        #print(f'df_true_class_calib :- {len(df_true_class_calib)}')




        true_class = test.iloc[:,-1]
        df_np = true_class.values
        df_true_class_test = torch.tensor(df_np, dtype=torch.int)
        #print(f'df_true_class_test :- {len(df_true_class_test)}')








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
            #print(f'avg_set_size:- {avg_set_size}')
            
            coverage_gap, coverage = coverage_gap_metric(conformal_set, df_true_class_test, args.alpha)
            #print(f'coverage_gap:- {coverage_gap}')
            #print(f'coverage:- {coverage}')

            
            avg_set_size_len_for_T_trials.append(avg_set_size)
            avg_coverage_gap_for_T_trials.append(coverage_gap)
            avg_coverage_for_T_trials.append(coverage)



        elif args.expt_no == 2:
            label = df_true_class_test
            indices_0 = torch.nonzero(label == 0).squeeze()
            indices_1 = torch.nonzero(label == 1).squeeze()
            indices_2 = torch.nonzero(label == 2).squeeze()
            indices_3 = torch.nonzero(label == 3).squeeze()
            indices_4 = torch.nonzero(label == 4).squeeze()
            indices_5 = torch.nonzero(label == 5).squeeze()
            indices_6 = torch.nonzero(label == 6).squeeze()
            indices_7 = torch.nonzero(label == 7).squeeze()


            benign_idx = torch.cat((indices_0, indices_1, indices_2, indices_3))
            malignant_idx = torch.cat((indices_4, indices_5, indices_6, indices_7))

            benign_conformal_prediction_set = conformal_set[benign_idx, :]
            malignant_conformal_prediction_set = conformal_set[malignant_idx, :]
            #print(f'benign_conformal_prediction_set :- {len(benign_conformal_prediction_set)}')
            #print(f'malignant_conformal_prediction_set :- {len(malignant_conformal_prediction_set)}')


            benign_avg_set_size_len = avg_set_size_metric(benign_conformal_prediction_set)
            malignant_avg_set_size_len = avg_set_size_metric(malignant_conformal_prediction_set)


            benign_true_class = df_true_class_test[benign_idx]
            malignant_true_class = df_true_class_test[malignant_idx]
            #print(f'benign_true_class :- {len(benign_true_class)}')
            #print(f'malignant_true_class :- {len(malignant_true_class)}')


            benign_coverage_gap, benign_coverage = coverage_gap_metric(benign_conformal_prediction_set, benign_true_class, args.alpha)

            malignant_coverage_gap, malignant_coverage = coverage_gap_metric(malignant_conformal_prediction_set, malignant_true_class, args.alpha)


            benign_avg_set_size_len_for_T_trials.append(benign_avg_set_size_len)
            benign_avg_coverage_gap_for_T_trials.append(benign_coverage_gap)
            benign_avg_coverage_for_T_trials.append(benign_coverage)


            malignant_avg_set_size_len_for_T_trials.append(malignant_avg_set_size_len)
            malignant_avg_coverage_gap_for_T_trials.append(malignant_coverage_gap)
            malignant_avg_coverage_for_T_trials.append(malignant_coverage)

        

        
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





    
    elif args.expt_no == 2:
        print()
        print()
        print(f'set_size :-')


        benign_avg_set_size_len_for_T_trials = np.array(benign_avg_set_size_len_for_T_trials)
        benign_average_set_size_len = np.mean(benign_avg_set_size_len_for_T_trials)
        benign_std_dev_set_size_len = np.std(benign_avg_set_size_len_for_T_trials, ddof=1)

        print()
        print(f"benign_average_set_size_len: {benign_average_set_size_len}")
        print(f"benign_std_dev_set_size_len: {benign_std_dev_set_size_len}")


        malignant_avg_set_size_len_for_T_trials = np.array(malignant_avg_set_size_len_for_T_trials)
        malignant_average_set_size_len = np.mean(malignant_avg_set_size_len_for_T_trials)
        malignant_std_dev_set_size_len = np.std(malignant_avg_set_size_len_for_T_trials, ddof=1)

        print()
        print(f"malignant_average_set_size_len: {malignant_average_set_size_len}")
        print(f"malignant_std_dev_set_size_len: {malignant_std_dev_set_size_len}")


        
        
        print()
        print()
        print(f'coverage:-')
        benign_avg_coverage_for_T_trials = np.array(benign_avg_coverage_for_T_trials)
        benign_average_coverage = np.mean(benign_avg_coverage_for_T_trials)
        benign_std_dev_coverage = np.std(benign_avg_coverage_for_T_trials, ddof=1)

        print()
        print(f"benign_average_coverage: {benign_average_coverage}")
        print(f"benign_std_dev_coverage: {benign_std_dev_coverage}")



        malignant_avg_coverage_for_T_trials = np.array(malignant_avg_coverage_for_T_trials)
        malignant_average_coverage = np.mean(malignant_avg_coverage_for_T_trials)
        malignant_std_dev_coverage = np.std(malignant_avg_coverage_for_T_trials, ddof=1)

        print()
        print(f"malignant_average_coverage: {malignant_average_coverage}")
        print(f"malignant_std_dev_coverage: {malignant_std_dev_coverage}")


    
    

            



    elif args.expt_no == 3:

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