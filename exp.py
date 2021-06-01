import numpy as np
import pandas as pd 
from fs_hawkes_beta import FSHawkesBeta
from scipy.stats import beta 
import argparse

def fs_hawkes_run(var_p: float, alpha_p: float, num_g: int, num_g_test: int, num_iter: int, data_set: str, run_mode: str):
    if data_set == "synthetic":
        df=pd.read_csv('./synthetic_data.csv',index_col=0)
        points_hawkes=[]
        for i in range(2):
            points_hawkes.append(list(df.iloc[i].values[~np.isnan(df.iloc[i].values)]))
        df=pd.read_csv('./synthetic_data_states_n.csv',index_col=0)
        states_n=[]
        for i in range(2):
            states_n.append(list(df.iloc[i].values[~np.isnan(df.iloc[i].values)].astype(int)))
        df=pd.read_csv('./synthetic_data_states.csv',index_col=0)
        states=list(df['0'])

        df=pd.read_csv('./synthetic_data_test.csv',index_col=0)
        points_hawkes_test=[]
        for i in range(2):
            points_hawkes_test.append(list(df.iloc[i].values[~np.isnan(df.iloc[i].values)]))
        df=pd.read_csv('./synthetic_data_states_n_test.csv',index_col=0)
        states_n_test=[]
        for i in range(2):
            states_n_test.append(list(df.iloc[i].values[~np.isnan(df.iloc[i].values)].astype(int)))
        df=pd.read_csv('./synthetic_data_states_test.csv',index_col=0)
        states_test=list(df['0'])

        beta_ab=np.array([[50,50,-2],[50,50,0]])
        T = 2000.0
        T_test = 2000.0
        T_phi = 6.0

    elif data_set == 'SCE':
        df=pd.read_csv('./earthquake_training.csv',index_col=0)
        points_hawkes=[]
        points_hawkes.append(df['timestamp(day)'].to_list())
        states_n=[]
        states_n.append(df['state'].to_list())
        states=df['state'].to_list()+[0]

        df=pd.read_csv('./earthquake_test.csv',index_col=0)
        points_hawkes_test=[]
        points_hawkes_test.append(df['timestamp(day)'].to_list())
        states_n_test=[]
        states_n_test.append(df['state'].to_list())
        states_test=df['state'].to_list()+[0]

        beta_ab=np.array([[1,100,0]]+[[50, 50, -5+1*i] for i in range(11)])
        T = np.ceil(max([points_hawkes[i][-1] for i in range(len(points_hawkes))]))
        T_test = np.ceil(max([points_hawkes_test[i][-1] for i in range(len(points_hawkes_test))]))
        T_phi = 10.0

    elif data_set == 'INTC':
        df=pd.read_csv('./INTC_training.csv',index_col=0)
        points_hawkes=[]
        points_hawkes.append(df.loc[df['EventType']==1,'Time'].to_list())
        points_hawkes.append(df.loc[df['EventType']==2,'Time'].to_list())
        states_n=[]
        states_n.append(df.loc[df['EventType']==1,'state'].to_list())
        states_n.append(df.loc[df['EventType']==2,'state'].to_list())
        states=df['state'].to_list()+[1]

        df=pd.read_csv('./INTC_test.csv',index_col=0)
        points_hawkes_test=[]
        points_hawkes_test.append(df.loc[df['EventType']==1,'Time'].to_list())
        points_hawkes_test.append(df.loc[df['EventType']==2,'Time'].to_list())
        states_n_test=[]
        states_n_test.append(df.loc[df['EventType']==1,'state'].to_list())
        states_n_test.append(df.loc[df['EventType']==2,'state'].to_list())
        states_test=df['state'].to_list()+[1]

        beta_ab=np.array([[50,50,-0.5+0.1*i] for i in range(3)])
        T = np.ceil(max([points_hawkes[i][-1] for i in range(len(points_hawkes))]))
        T_test = np.ceil(max([points_hawkes_test[i][-1] for i in range(len(points_hawkes_test))]))
        T_phi = 1.0

    else: 
        print("data_set not supported!")

    num_of_state = len(np.unique(states))
    num_of_dims = len(points_hawkes)
    num_of_basis = len(beta_ab)
    fs_model = FSHawkesBeta(num_of_state,num_of_dims,num_of_basis)
    fs_model.set_hawkes_hyperparameters(beta_ab,T_phi)

    if run_mode == "gibbs":
        lamda_ub_estimated_list_gibbs,W_estimated_list_gibbs,P_estimated_list_gibbs,logl_list_gibbs,logl_test_list_gibbs\
        =fs_model.Gibbs(points_hawkes,states,states_n,points_hawkes_test,states_test,states_n_test,\
                 T,T_test,var_p,alpha_p,num_g,num_g_test,num_iter)

        lamda_ub_estimated_mean_gibbs=np.mean(lamda_ub_estimated_list_gibbs[-100:],axis=0)
        W_estimated_mean_gibbs=np.mean(W_estimated_list_gibbs[-100:],axis=0)
        P_estimated_mean_gibbs=np.mean(P_estimated_list_gibbs[-100:],axis=0)

        print("Estimated lambda ub (mean)")
        print(lamda_ub_estimated_mean_gibbs)
        print("="*56)
        print("Estimated base activation (mean)")
        print(W_estimated_mean_gibbs[:,:,0])
        print("="*56)
        print("Estimated weight (mean)")
        print(W_estimated_mean_gibbs[:,:,1:])
        print("Estimated state transition matrix (mean)")
        print(P_estimated_mean_gibbs)

    elif run_mode == 'mf':
        alpha_estimated_mf,mean_W_estimated_mf,cov_W_estimated_mf,dir_alpha_estimated_mf,logl_list_mf,logl_test_list_mf\
        =fs_model.MeanField(points_hawkes,states,states_n,points_hawkes_test,states_test,states_n_test,\
                 T,T_test,var_p,alpha_p,num_g,num_g_test,num_iter)

        lamda_ub_estimated_mean_mf=alpha_estimated_mf/T
        P_estimated_mean_mf=np.zeros((num_of_dims,num_of_state,num_of_state))
        for i in range(num_of_dims):
            for j in range(num_of_state):
                P_estimated_mean_mf[i][j]=dir_alpha_estimated_mf[i][j]/sum(dir_alpha_estimated_mf[i][j])

        print("Estimated lambda ub (mean)")
        print(lamda_ub_estimated_mean_mf)
        print("="*56)
        print("Estimated base activation (mean)")
        print(mean_W_estimated_mf[:,:,0])
        print("="*56)
        print("Estimated weight (mean)")
        print(mean_W_estimated_mf[:,:,1:])
        print("Estimated state transition matrix (mean)")
        print(P_estimated_mean_mf)

    else: 
        print("run_mode not supported!")

                                                               



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--var_p",
                        dest="var_p",
                        required=True,
                        type=float,
                        help="variance of Gaussian prior distribution")
    parser.add_argument("-a", "--alpha_p",
                        dest="alpha_p",
                        required=True,
                        type=float,
                        help="parameter of Dirichlet prior distribution")
    parser.add_argument("-ng", "--num_g",
                        dest="num_g",
                        required=True,
                        type=int,
                        help="number of grid in Gibbs sampler / number of \
                        Gaussian quadrature nodes for each state-segment in mean-field (training data)")
    parser.add_argument("-ngt", "--num_g_test",
                        dest="num_g_test",
                        required=True,
                        type=int,
                        help="number of grid in Gibbs sampler / number of \
                        Gaussian quadrature nodes for each state-segment in mean-field (test data)")
    parser.add_argument("-niter", "--num_iter",
                        dest="num_iter",
                        required=True,
                        type=int,
                        help="number of iterations")
    parser.add_argument("-d", "--data_set",
                        dest="data_set",
                        required=True,
                        type=str,
                        help="dataset",
                        choices=['synthetic', 'SCE', 'INTC'])
    parser.add_argument("-m", "--run_mode",
                        dest="run_mode",
                        required=True,
                        type=str,
                        help="algorithm",
                        choices=['gibbs', 'mf'])
    args = parser.parse_args()
 
    fs_hawkes_run(args.var_p,args.alpha_p,args.num_g,args.num_g_test,args.num_iter,args.data_set,args.run_mode)
    
    print("Done!")