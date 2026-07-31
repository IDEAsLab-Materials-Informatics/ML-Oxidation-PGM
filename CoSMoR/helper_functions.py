"""script with any user-defined functions or classes that are external to 'CoSMoR'.
A common use-case for this script would be to store functions required for creating alloy features.
"""

# import libraries
import numpy as np
import pandas as pd
import re
import math
import os
from core_functions import get_comp_dict

SEP = os.sep

# load element database

db_element = pd.read_csv("db_element.csv",encoding="latin-1")
db_element = db_element.set_index('Symbol') #set 'Symbol' column as index



# -----------------------------------------------
def get_comp_avg(alloy_name_list, feat_key):
    """creates composition-weighted average property for alloys

    Args:
        alloy_name_list (list): list of alloy names as string
        el_prop (str): element property to use

    Returns:
        feat_val_list (list): list of composition-weighted average feature values
    """
    
    feat_val_list = []
    
    for alloy in alloy_name_list:
        dict_alloy_comp = get_comp_dict(alloy)
        el_list = dict_alloy_comp["el_list"]
        el_prop_list = [db_element.loc[el][feat_key] for el in el_list]
        c_el_list = dict_alloy_comp["el_at_frac_list"]
        
        avg_feat = 0
        
        for (c_el, el_prop) in zip(c_el_list, el_prop_list):
            avg_feat += (c_el * el_prop)
            
        feat_val_list.append(avg_feat)
        
    return (feat_val_list)


def get_alloy_comp(alloy_name):

    """
    Input: alloy_name e.g. 'AlCu2Fe0.5Co0.5'
    Output: composition array with element alloy_name and atomic fraction e.g. [[Al, 0.25],[Cu, 0.5],[Fe, 0.125],[Co, 0.125]]
            Column 0 of output array hold elements present, Column 1 holds atomic fraction of each element

    """


    # function to count uppercase alphabets in a word
    def get_count_uppercase(word):


        count=0;
        word=str(word);
        
        for i in range(0,len(word)):

            if word[i].isupper(): count=count+1;


        return(count)


    # function to convert composition array values into atomic fractions
    def get_at_fraction(comp_array):

        
        total = np.sum(comp_array, axis=0)[1];      #total of 2nd column
        comp_array[:,1] = comp_array[:,1]/total;

        
        return comp_array
    

    exclude = set("() {}[]''/,;:?\|~`@#$%&^*-_");                       #to be removed from hea alloy_name
    alloy_name = ''.join(ch for ch in alloy_name if ch not in exclude); #excluding characters defined in exclude set
    n_element = get_count_uppercase(alloy_name);                        #no. of elements=no. of uppercase letters
    comp_array = np.empty((n_element,2)).astype('object');              #creating composition array to hea composition: no. of rows=no. of elements
    R=-1; C=0;                                                          #initiating row and column count
    
    for j in range(0,len(alloy_name)):                #consider each letter at a time
        
        if alloy_name[j].isupper() == True:               #if uppercase

            R=R+1;                                          #increases row number
            C=0;                                            #column no. becomes 0 i.e first column
            comp_array[R,C] = str(alloy_name[j]);           #alphabet stored in first column

            if (j+1)<len(alloy_name):                       #if not the last alphabet

                if alloy_name[j+1].isupper() == True:       #if next alphabet is uppercase

                    C=1; comp_array[R,C] = 1;               #store value of 'one' in second column
                    
        elif alloy_name[j].islower() == True:                                 #if lowercase
            
            C=0;                                                                #column becomes zero
            comp_array[R,C]=str(str(comp_array[R,C])+str(alloy_name[j]));       #concatenate to alloy_name of element in first column
            
            if (j+1)<len(alloy_name):                                           #if not the last alphabet

                if alloy_name[j+1].isupper() == True:                           #if next alphabet is uppercase

                    C=1; comp_array[R,C] = 1;                                   #store value of 'one' in second column
                    
        else:                                                       #if neither upper nor lowercase

            C=1;                                                    #column becomes 1 i.e. second column

            if alloy_name[j-1].isalpha() == True:                   #if previous letter is alphabet

                comp_array[R,C]=alloy_name[j];                      #store value in second column

            else:                                                           #if previous letter not an alphabet

                comp_array[R,C]=str(comp_array[R,C])+str(alloy_name[j]);    #concatenate with value stored in second column

    if alloy_name[-1].isalpha() == True:        #if the last letter of alloy_name is alphabet

        comp_array[R,1] = 1;                    #make the last row, second column entry = 1
    
    #Convert all second column value (compositions) to float numbers
    for R in range (0,comp_array.shape[0]):
        
        comp_array[R,1] = float(comp_array[R,1]);
    
    comp_at_array = get_at_fraction(comp_array); #call function to convert comp. values into atomic conc.
    

    return comp_at_array;

# -----------------------------------------------
def get_asymmetry(alloy_name_list, feat_key):
    """creates asymmetry properties for alloys

    Args:
        alloy_name_list (list): list of alloy names as string
        el_prop (str): element property to use

    Returns:
        feat_val_list (list): list of asymmetry feature values
    """
    
    feat_val_list = []
    avg_feat_list = get_comp_avg(alloy_name_list, feat_key)
    
    for (alloy, avg_feat) in zip(alloy_name_list, avg_feat_list):
        dict_alloy_comp = get_comp_dict(alloy)
        el_list = dict_alloy_comp["el_list"]
        
        el_prop_list = [db_element.loc[el][feat_key] for el in el_list]
        
        c_el_list = dict_alloy_comp["el_at_frac_list"]

        asymm_feat_sum = 0  # summation term inside the square root

        for (c_el, el_prop) in zip(c_el_list, el_prop_list):
            asymm_feat_sum += (c_el * (1-(el_prop/avg_feat))**2)

        asymm_feat = (asymm_feat_sum)**(0.5)
        
        feat_val_list.append(asymm_feat)
    
    return (feat_val_list)
 
 
 
 
 
def get_miedema_enthalpy(alloy_name, state="liquid", show_calc=False, model="R"):
    
    """
    Inputs:
        'alloy_name': eg: 'CuFe2Au1Zn2'
        'state': 'solid' or 'liquid'
        'show_calc': boolean: True or False ; If True, calculations are displayed
        'model': 'R' for regular, 'NR' for non-regular

    Outputs:
        'H_chem_alloy': Chemical Enthalpy of Mixing of alloy
        'H_el_alloy': Elastic Enthalpy of Mixing of alloy
        'H_mix_alloy': Total Enthalpy of Mixing of alloy: H_chem + H_el
        
    """
    
    db_element = pd.read_csv(f"db_element.csv", encoding='latin-1')
    db_element = db_element.set_index('Symbol') #set 'Symbol' column as index

    comp_array = get_alloy_comp(alloy_name); 
    n_element = comp_array.shape[0]; 
    
    n_binaries = int(
            math.factorial(n_element)/
            (math.factorial(n_element-2)*math.factorial(2))
            ); # calculating no. of unique binary systems in alloy
    
    H_mix_alloy_array = np.zeros(n_binaries); # array to store H_mix of binaries; shape=(no. of binaries,)
    H_chem_alloy_array = np.zeros(n_binaries); # array to store H_chem of binaries; shape=(no. of binaries,)
    H_el_alloy_array = np.zeros(n_binaries); # array to store H_elastic of binaries; shape=(no. of binaries,)

    C = comp_array[:,1]; # C = col1 of composition array; shape=(n_element, 1)
    
    # printing calculations if 'show_calc' boolean is 'True'
    if show_calc == True:
        
        print('alloy Composition:\n',comp_array);
        print('\nNo. of elements=',n_element);
        print('No. of binaries=',n_binaries);
    
    count = 0
    
    
    # loop to iterate over binary systems and calculate one binary enthalpy in each run; runs for 'no. of elements-1' times
    for j in range(0,n_element-1):              # j represents first element in binary; say A
        
        # loop to create
        for k in range(j+1,n_element):          # k represents second element in binary; say B: j=0-> k=1,2,3,4 :: j=1->k=2,3,4 ::j =2->

            # Calculating delta_Hmix of each equiatomic binary
            el_A = comp_array[j,0];

            if model == 'R': cA = 0.5;      # if 'regular model', then equiatomic system: i.e. cA=0.5=cB
            if model == 'NR': cA = C[j];    # if 'non-regular model', then equiatomic system: i.e. cA=cB=actual conc in alloy
            
            # collecting el_A element data from 'db_element' database
            Vm_A = db_element.loc[el_A,'V_m']; 
            w_fn_A = db_element.loc[el_A,'Work_Function']; 
            nWS_A = db_element.loc[el_A,'n_WS']; 
            K_A = db_element.loc[el_A,'E_GPa'];
            G_A = db_element.loc[el_A,'G_GPa'];
            type_A = db_element.loc[el_A,'Type']; 

            el_B = comp_array[k,0];

            if model == 'R': cB = 0.5;      # if 'regular model', then equiatomic system: i.e. cA=0.5=cB
            if model == 'NR': cB = C[k];    # if 'non-regular model', then equiatomic system: i.e. cA=cB=actual conc in alloy
            
            # collecting el_B element data from 'db_element' database
            Vm_B = db_element.loc[el_B,'V_m'];
            w_fn_B = db_element.loc[el_B,'Work_Function']; 
            nWS_B = db_element.loc[el_B,'n_WS']; 
            K_B = db_element.loc[el_B,'E_GPa'];
            G_B = db_element.loc[el_B,'G_GPa'];
            type_B = db_element.loc[el_B,'Type']; 

            cA_s = cA*(Vm_A**2/3)/(cA*(Vm_A**2/3)+0.5*(Vm_B**2/3)); # surface concentration of el_A
            cB_s = cB*(Vm_B**2/3)/(cB*(Vm_A**2/3)+0.5*(Vm_B**2/3)); # surface concentration of el_B

            del_w_fn = w_fn_A - w_fn_B; # diff in work function
            del_nWS1_3 = nWS_A**(1/3) - nWS_B**(1/3); # delta(nWS^1_3)
            nWS_1_3_avg = (1/2)*(nWS_A**(-1/3)+nWS_B**(-1/3)); # average((nWS^-1_3)

            Vm_A_corr = 1.5*cA_s*(Vm_A**(2/3))*(w_fn_A-w_fn_B)*((1/nWS_B)-(1/nWS_A))/(2*nWS_1_3_avg); # volume correction of el_A
            Vm_B_corr = 1.5*cB_s*(Vm_B**(2/3))*(w_fn_B-w_fn_A)*((1/nWS_A)-(1/nWS_B))/(2*nWS_1_3_avg); # volume correction of el_B
            Vm_A_alloy = Vm_A + Vm_A_corr; # corrected volume of el_A
            Vm_B_alloy = Vm_B + Vm_B_corr; # corrected volume of el_B

            # Selecting P,Q,R based on type of elements and alloy_phase
            if (db_element.loc[el_A,'Type']=='transition'\
                and db_element.loc[el_B,'Type']=='transition'):

                P=14.2; Q=9.4*P; R=0;

            if (db_element.loc[el_A,'Type']=='non_transition'\
                and db_element.loc[el_B,'Type']=='non_transition'):

                P=10.7; Q=9.4*P; R=0;

            if (db_element.loc[el_A,'Type']!= db_element.loc[el_B,'Type']):
                P=12.35; Q=9.4*P;

                R_P_A = db_element.loc[comp_array[j,0],'R_P'];
                R_P_B = db_element.loc[comp_array[k,0],'R_P'];
                R_P = R_P_A*R_P_B;

                if state=='solid': R = 1*R_P*P; 
                else: R = 0.73*R_P*P;
            
            
            tau = (1/nWS_1_3_avg)*(-P*(del_w_fn**2)+Q*(del_nWS1_3**2)-R); # tau parameter
            H_chem_AB = (cA*cB)*(cB_s*(Vm_A_alloy**(2/3))+cA_s*(Vm_B_alloy**(2/3)))*tau; # calculating H_chemical

            H_el_AinB = 2*K_A*G_B*((Vm_A_alloy-Vm_B_alloy)**2)/(3*K_A*Vm_B_alloy + 4*G_B*Vm_A_alloy); # H_elastic A in B
            H_el_BinA = 2*K_B*G_A*((Vm_A_alloy-Vm_B_alloy)**2)/(3*K_B*Vm_A_alloy + 4*G_A*Vm_B_alloy); # H_elastic B in A
            H_el_AB = cA*cB*(cB*H_el_AinB + cA*H_el_BinA); # # H_elastic of A-B binary
            
            H_mix_AB = H_chem_AB + H_el_AB; # H_mix of binary = H_chemical + H_elastic
            
            # If 'regular' model is selected; then enthalpies scaled by 4*cA*cB factor where cA,cB are actual conc. in alloy
            if model == 'R':
                
                cA_alloy = C[j]; cB_alloy = C[k]; #actual at. conc. of A and B
                H_chem_alloy_array[count] = 4*cA_alloy*cB_alloy*H_chem_AB; # added to 'H_chemical' array
                H_el_alloy_array[count] = 4*cA_alloy*cB_alloy*H_el_AB; # added to 'H_elastic' array
                H_mix_alloy_array[count] = 4*cA_alloy*cB_alloy*H_mix_AB; # added to 'H_mix' array
            
            # If 'non-regular' model is selected; no scaling is required
            if model == 'NR':
                
                H_chem_alloy_array[count] = H_chem_AB; # added to 'H_chemical' array
                H_el_alloy_array[count] = H_el_AB; # added to 'H_elastic' array
                H_mix_alloy_array[count] = H_mix_AB; # added to 'H_mix' array
            
            # Printing calculations if 'show_calc' input in function call is 'True'
            if (show_calc==True):

                Bold_s='\033[1m'; Bold_f='\033[0m';
                print(Bold_s+'\nBinary No.',(count+1),'('+el_A+el_B+'):'+Bold_f);
                print('A=%s; cA=%.2f; Vm(A)=%.2f; Work_F(A)=%.2f; nWS(A)=%.2f; K(A)=%.2f; G(A)=%.2f; Type(A)=%s'
                      %(el_A, cA, Vm_A, w_fn_A, nWS_A, K_A, G_A, type_A));

                print('B=%s; cB=%.2f; Vm(B)=%.2f; Work_F(B)=%.2f; nWS(B)=%.2f; K(B)=%.2f; G(B)=%.2f; Type(B)=%s'
                      %(el_B, cB, Vm_B, w_fn_B, nWS_B, K_B, G_B, type_B));

                print('\tcA(s)=%.3f; cB(s)=%.3f; Vm(A)_alloy=%.3f; Vm(B)_alloy=%.3f'
                      %(cA_s, cB_s, Vm_A_alloy, Vm_B_alloy));

                print(Bold_s+'\tChemical Enthalpy:'+Bold_f);
                print('\tdelta(Work_F)=%.2f; delta(nWS^(1/3))=%.2f; nWS^(-1/3)avg=%.2f'
                      %(del_w_fn, del_nWS1_3, nWS_1_3_avg));

                print('\tP=%.2f; R=%.2f; Q=%.2f; Tau=%.3f'%(P, R, Q, tau))
                print(Bold_s+'\t\tH_chem('+el_A+el_B+')= %.3f kJ/mol'%(H_chem_AB)+Bold_f);

                print(Bold_s+'\tElastic Enthalpy:'+Bold_f)
                print('\tH_el(A in B)=%.3f; H_el(B in A)=%.3f'%(H_el_AinB, H_el_BinA));        
                print(Bold_s+'\t\tH_el('+el_A+el_B+')= %.3f'%H_el_AB+Bold_f);

                print(Bold_s+'\n\t\t\t\tdelta_H_mix('+el_A+el_B+')= %.3f'%H_mix_AB+Bold_f);
            
            count+=1;

    # alloy enthalpies = mean of all binary enthalpies
    H_chem_alloy = np.around(np.sum(H_chem_alloy_array),2); 
    H_el_alloy = np.around(np.sum(H_el_alloy_array),2);
    H_mix_alloy = np.around(np.sum(H_mix_alloy_array),2);
    
    if show_calc == True:

        print('\nChemical Enthalpy of mixing of alloy(%s)=%.3f kJ/mol'%(alloy_name,H_chem_alloy));
        print('Elastic Enthalpy of mixing of alloy(%s)=%.3f kJ/mol'%(alloy_name,H_el_alloy));
        print('Enthalpy of mixing of alloy(%s)=%.3f kJ/mol\n'%(alloy_name,H_mix_alloy));

    
    return [H_chem_alloy, H_el_alloy, H_mix_alloy]
