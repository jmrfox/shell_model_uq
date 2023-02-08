#
#   computes many samples of E2 transitions using BIGSTICK
#   Fox 2019
#

from shmuq_mod import *

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filename_csv',default='sd_E2_processed_v2.csv')
    parser.add_argument('--output_filename_csv',default='sd_E2_usdb_complete.csv')
    parser.add_argument('--sample_number',default=None)
    args = parser.parse_args()

    # int_name is a string set in the SHMUQ module, i.e. = 'usdb'
    sample_number = args.sample_number
    if sample_number is None:
        sample_int_name = int_name
    else:
        sample_int_name = make_sample_interaction(int_name,sample_number,milcoms_filename)
    print('Interaction = ',sample_int_name)

    #if int_name.endswith('.int'):
    #    int_name = int_name.strip('.int')
    df = pd.read_csv(args.input_filename_csv)
    df = df[df['Include']==True]
    df['Mth_p'] = 0.0
    df['Mth_n'] = 0.0

    nuclei = np.array(df[['A','Element','Z','N']].drop_duplicates())

    i_trans = 0
    for nuc_entry in nuclei:
        if verbose: print('nuc_entry',nuc_entry)
        transitions = df[(df['A']==nuc_entry[0]) & (df['Element']==nuc_entry[1])]   #'transitions' is a subset of dataframe

        nuc = get_nucleus(transitions.iloc[0])

        run_name = make_bigstick_inputs(sample_int_name,nuc,'d',twoJz=nuc.twoJ)
        run_bigstick(run_name)

        run_name_gs_p = make_genstrength_inputs(run_name,sample_int_name,nuc,opme_name_E2_p)
        run_name_gs_n = make_genstrength_inputs(run_name,sample_int_name,nuc,opme_name_E2_n)
        run_genstrength(run_name_gs_p)
        run_genstrength(run_name_gs_n)

        state_index_list = state_indices(transitions,sample_int_name,run_name)
        print('State index list',state_index_list)

        if (nuc.Zv>0) and ((max_Z - core_Z - nuc.Zv) > 0):
            transitions = parse_strength_file_by_idx(run_name_gs_p,state_index_list,transitions,'Mth_p')
            df.iloc[i_trans:i_trans+len(transitions)]['Mth_p'] = transitions['Mth_p']
        else:
            if verbose: print('NO PROTON PART!')
            df.iloc[i_trans:i_trans+len(transitions)]['Mth_p'] = 0.

        if (nuc.Nv>0) and ((max_N - core_N - nuc.Nv) > 0):
            transitions = parse_strength_file_by_idx(run_name_gs_n,state_index_list,transitions,'Mth_n')
            df.iloc[i_trans:i_trans+len(transitions)]['Mth_n'] = transitions['Mth_n']
        else:
            if verbose: print('NO NEUTRON PART!')
            df.iloc[i_trans:i_trans+len(transitions)]['Mth_n'] = 0.

        if verbose: print(transitions)
        i_trans += len(transitions)

    df.to_csv(args.output_filename_csv)

    print("Done." + f'Results written to {args.output_filename_csv}')




