'''
This is the file where we define the cases we want to use for training or testing the model.

Standard way to introduce a new case:
# NOTE: New case
# Replace 
# 1. *Case_name* with the name of the case
# 2. *./data/datafile.h5* with the path to the data file
# 3. *Title for the case* with the title of the case
INPUT_TURB_FILES[f'Case_name'] = f'./data/datafile.h5']
TURB_CASES.append(f'Case_name')
DATASET_PLOT_TITLE[f'Case_name'] = f'Title for the case'
'''

# NOTE: BFM_WM_v2: this registry also contains every case added in the BFM_WM error-model/SNGP fork
# (A-airfoil 2M/10M full-2D-field sets and their region splits, transition_JHU and ph_B full-2D fields,
# NASA hump pressure-gradient regions hump_*, and the a-posteriori LES sets aposteriori_*).
# Shipped in ./data: the training data (the hump_* regions and the laminar A-airfoil sets
# aairfoil_2M_full2Dfield_{down0p02_up0p8,down0p005_up1}_laminar_{ZPG,APG,FPG}; the down0p005_up1 keys are
# added by BFM_WM_v2) and the lam-turb test set aairfoil_2M_full2Dfield_down0_up1. The other added keys
# (aposteriori_*, aairfoil_10M*, the remaining aairfoil_2M_full2Dfield* and aairfoil_2M_{laminar,
# transition,turbulent}, transition_JHU_full2Dfield*, ph_B_full2Dfield*) point to files that are not
# in this repository; place the files in ./data (or pass explicit paths in the test TOMLs) to use them.

import numpy as np

# NOTE: High-fidelity data: case names and corresponding input files
TURB_CASES = [
              # High Reynolds number ZPG cases
              'CH', 'SYN', 'PIPE',
              # TBL cases
              'naca_0012', 'naca_0025',
              'backstep', 'axissym_BL', 'ph_B',  'bend', 'convdiv',
              'hump', 'smoothramp', 'round_step']

# NOTE: File extension for the input files
EXT = 'h5'

# NOTE: TBL has different ramp angles; will be updated by TBL
INPUT_TURB_FILES = dict(zip(TURB_CASES, [f'./data/{case}_data.{EXT}' for case in TURB_CASES]))
DATASET_PLOT_TITLE = dict(zip(TURB_CASES, [r'Channel flow $Re_{\tau}:550-4200$', 'Synthetic (log law) data (up to $Re_{\tau}=100,000$)', 'Pipe (up to $Re_{\tau}=12,000$)', 'NACA0012 Re=400K', 'NACA0025', 
                                        '2D Backward Step (Driver et al. 1985)', 'Axissymetric Boundary Layer (Driver et al. 1991)', 'Periodic Hill (Balakumar 2015)', 'Bended Boundary Layer (Smits et al. 1979)', 'Convergent and Divergent Channel (Laval et al. 2009)',
                                        'NASA Hump (Uzun et al. 2017)', 'Smooth Ramp (Uzun et al. 2024)', 'Round Step DLR']))

# NOTE: Spinning 3D TBL by Driver et al. (1987)
casename = 'spinning_Driver'
SUBCASES = ['as1', 'cs0', 'cs1', 'ds0', 'ds1']
for case in SUBCASES:
    INPUT_TURB_FILES[f'{casename}_{case}'] = f'./data/{casename}_{case}_data.{EXT}'
    TURB_CASES.append(f'{casename}_{case}')
    DATASET_PLOT_TITLE[f'{casename}_{case}'] = f'{casename}_{case}'

# NOTE: Transition TBL by Coupland
casename = 'transition_Coupland'
SUBCASES = ['t3am', 't3a', 't3b' , 't3c1', 't3c2', 't3c3', 't3c4', 't3c5']
for case in SUBCASES:
    INPUT_TURB_FILES[f'{casename}_{case}'] = f'./data/{casename}_{case}_data.{EXT}'
    TURB_CASES.append(f'{casename}_{case}')
    DATASET_PLOT_TITLE[f'{casename}_{case}'] = f'{casename}_{case}'

# NOTE: Strained TBL
casename = 'strained_TBL'
Z_LIST = [-150, -100, 0]
for z in Z_LIST:
    INPUT_TURB_FILES[f'{casename}_z{z}'] = f'./data/{casename}_z{z}_data.{EXT}'
    TURB_CASES.append(f'{casename}_z{z}')
    DATASET_PLOT_TITLE[f'{casename}_z{z}'] = f'{casename} z={z}'

# NOTE: Psuedo-Ekman by Spalart 1988
INPUT_TURB_FILES[f'Ekman'] = f'./data/Ekman_data.{EXT}'
TURB_CASES.append(f'Ekman')
DATASET_PLOT_TITLE[f'Ekman'] = f'Pseudo-Ekman by Spalart (1988)'

# NOTE: Concave TBL by Johnson et al. (1988)
INPUT_TURB_FILES[f'Concave_J'] = f'./data/Concave_Johnson_data.{EXT}'
TURB_CASES.append(f'Concave_J')
DATASET_PLOT_TITLE[f'Concave_J'] = f'Concave TBL by Johnson et al. (1988)'

# NOTE: APG TBL by Gasser et al.
INPUT_TURB_FILES[f'APG_Gasser'] = f'./data/APG_Gasser_data.{EXT}'
TURB_CASES.append(f'APG_Gasser')
DATASET_PLOT_TITLE[f'APG_Gasser'] = f'APG Gasser et al.'

# NOTE: Backstep step with 6 degree expansion angle
INPUT_TURB_FILES[f'backstep_inc'] = f'./data/backstep_inclined_data.{EXT}'
TURB_CASES.append(f'backstep_inc')
DATASET_PLOT_TITLE[f'backstep_inc'] = f'2D Backward Step with 6 degree inclination (Driver et al. 1985)'

# NOTE: Swept wing (35 degree yaw) data
INPUT_TURB_FILES[f'swept_wing'] = f'./data/swept_wing_data.h5'
TURB_CASES.append(f'swept_wing')
DATASET_PLOT_TITLE[f'swept_wing'] = f'Swept wing (Van den Berg et al. 1975)'

# NOTE: Duct
INPUT_TURB_FILES[f'duct'] = f'./data/DUCT_data.h5'
TURB_CASES.append(f'duct')
DATASET_PLOT_TITLE[f'duct'] = f'Duct (Pirozzo et al. 2018)'

# NOTE: diffuser
INPUT_TURB_FILES[f'diffuser'] = f'./data/diffuser_data.h5'
TURB_CASES.append(f'diffuser')
DATASET_PLOT_TITLE[f'diffuser'] = f'Cherry Diffuser (Ohlsson et al. 2010)'

# NOTE: Channel with rotation
INPUT_TURB_FILES[f'CH_rot'] = f'./data/CH_rot_data.h5'
TURB_CASES.append(f'CH_rot')
DATASET_PLOT_TITLE[f'CH_rot'] = f'Channel with Rotation'

# NOTE: Our TBL data
TBL_ANGLES = [-4,-3,-2,-1,5,10,15,20]
for angle in TBL_ANGLES:
    INPUT_TURB_FILES[f'TBL_{angle}'] = f'./data/TBL_{angle}_data.{EXT}'
    TURB_CASES.append(f'TBL_{angle}')
    DATASET_PLOT_TITLE[f'{angle}'] = f'Turbulent Boundary Layer, Ramp Angle: {angle}°'
    DATASET_PLOT_TITLE[f'TBL_{angle}'] = f'Turbulent Boundary Layer, Ramp Angle: {angle}°'
# NOTE: APG data 
SUBCASES = ['b1n', 'b2n', 'm13n', 'm16n', 'm18n']
for subcase in SUBCASES:
    INPUT_TURB_FILES[f'apg_{subcase}'] = f'./data/apg_{subcase}_data.{EXT}'
    TURB_CASES.append(f'apg_{subcase}')
    DATASET_PLOT_TITLE[f'apg_{subcase}'] = f'Mild Adverse Pressure Gradient, Subcase: {subcase}'
# NOTE: aairfoil with different Reynolds numbers
RE = [2, 10]
for re in RE:
    INPUT_TURB_FILES[f'aairfoil_{re}M'] = f'./data/aairfoil_{re}M_data.{EXT}'
    TURB_CASES.append(f'aairfoil_{re}M')
    DATASET_PLOT_TITLE[f'aairfoil_{re}M'] = f'A-Airfoil, Reynolds number: {re}M'
RE = 2
POS = ['laminar', 'transition', 'turbulent']
for pos in POS:
    INPUT_TURB_FILES[f'aairfoil_{RE}M_{pos}'] = f'./data/aairfoil_{RE}M_{pos}_data.{EXT}'
    TURB_CASES.append(f'aairfoil_{RE}M_{pos}')
    DATASET_PLOT_TITLE[f'aairfoil_{RE}M_{pos}'] = f'A-Airfoil, Reynolds number: {RE}M, {pos} region'

# New Full 2D field data for aairfoil at Re=2M and 10M
RE = [2, 10]
for re in RE:
    INPUT_TURB_FILES[f'aairfoil_{re}M_full2Dfield'] = f'./data/aairfoil_{re}M_full2Dfield.{EXT}'
    TURB_CASES.append(f'aairfoil_{re}M_full2Dfield')
    DATASET_PLOT_TITLE[f'aairfoil_{re}M_full2Dfield'] = f'A-Airfoil, Reynolds number: {re}M, Full 2D field'

# Also split the full 2D field data into laminar, transition, and turbulent regions based on x/c location
POS = ['laminar', 'transition', 'turbulent']
for re in RE:    
    for pos in POS:
        INPUT_TURB_FILES[f'aairfoil_{re}M_full2Dfield_{pos}'] = f'./data/aairfoil_{re}M_full2Dfield_{pos}.{EXT}'
        TURB_CASES.append(f'aairfoil_{re}M_full2Dfield_{pos}')
        DATASET_PLOT_TITLE[f'aairfoil_{re}M_full2Dfield_{pos}'] = f'A-Airfoil, Reynolds number: {re}M, Full 2D field, {pos} region'

# Full 2D field with wall-normal sampling range specified
for re in RE:
    INPUT_TURB_FILES[f'aairfoil_{re}M_full2Dfield_down0p02_up0p8'] = f'./data/aairfoil_{re}M_full2Dfield_down0p02_up0p8.{EXT}'
    TURB_CASES.append(f'aairfoil_{re}M_full2Dfield_down0p02_up0p8')
    DATASET_PLOT_TITLE[f'aairfoil_{re}M_full2Dfield_down0p02_up0p8'] = f'A-Airfoil, Reynolds number: {re}M, Full 2D field'

for re in RE:    
    for pos in POS:
        INPUT_TURB_FILES[f'aairfoil_{re}M_full2Dfield_down0p02_up0p8_{pos}'] = f'./data/aairfoil_{re}M_full2Dfield_down0p02_up0p8_{pos}.{EXT}'
        TURB_CASES.append(f'aairfoil_{re}M_full2Dfield_down0p02_up0p8_{pos}')
        DATASET_PLOT_TITLE[f'aairfoil_{re}M_full2Dfield_down0p02_up0p8_{pos}'] = f'A-Airfoil, Reynolds number: {re}M, Full 2D field, {pos} region'

REG = ['APG', 'FPG', 'ZPG']
for re in RE:    
    for pos in POS:
        for reg in REG:
            INPUT_TURB_FILES[f'aairfoil_{re}M_full2Dfield_down0p02_up0p8_{pos}_{reg}'] = f'./data/aairfoil_{re}M_full2Dfield_down0p02_up0p8_{pos}_{reg}.{EXT}'
            TURB_CASES.append(f'aairfoil_{re}M_full2Dfield_down0p02_up0p8_{pos}_{reg}')
            DATASET_PLOT_TITLE[f'aairfoil_{re}M_full2Dfield_down0p02_up0p8_{pos}_{reg}'] = f'A-Airfoil, Reynolds number: {re}M, Full 2D field, {pos}, {reg} region'

# Full 2D field with wall-normal sampling range specified
for re in RE:
    INPUT_TURB_FILES[f'aairfoil_{re}M_full2Dfield_down0_up1'] = f'./data/aairfoil_{re}M_full2Dfield_down0_up1.{EXT}'
    TURB_CASES.append(f'aairfoil_{re}M_full2Dfield_down0_up1')
    DATASET_PLOT_TITLE[f'aairfoil_{re}M_full2Dfield_down0_up1'] = f'A-Airfoil, Reynolds number: {re}M, Full 2D field'

# Filtered Cf: Full 2D field with wall-normal sampling range specified
for re in RE:
        INPUT_TURB_FILES[f'aairfoil_{re}M_full2Dfield_down0_up1_cfgauss_w5_tr3_x0p02to0p98'] = f'./data/aairfoil_{re}M_full2Dfield_down0_up1_cfgauss_w5_tr3_x0p02to0p98.{EXT}'
        TURB_CASES.append(f'aairfoil_{re}M_full2Dfield_down0_up1_cfgauss_w5_tr3_x0p02to0p98')
        DATASET_PLOT_TITLE[f'aairfoil_{re}M_full2Dfield_down0_up1_cfgauss_w5_tr3_x0p02to0p98'] = f'A-Airfoil, Reynolds number: {re}M, Full 2D field'

for re in RE:
        INPUT_TURB_FILES[f'aairfoil_{re}M_full2Dfield_down0_up1_cfgauss_w30_tr3_xalltoall'] = f'./data/aairfoil_{re}M_full2Dfield_down0_up1_cfgauss_w30_tr3_xalltoall.{EXT}'
        TURB_CASES.append(f'aairfoil_{re}M_full2Dfield_down0_up1_cfgauss_w30_tr3_xalltoall')
        DATASET_PLOT_TITLE[f'aairfoil_{re}M_full2Dfield_down0_up1_cfgauss_w30_tr3_xalltoall'] = f'A-Airfoil, Reynolds number: {re}M, Full 2D field'

# BFM_WM_v2: laminar A-airfoil sets from n1/delta 0.005 to 1, the additional training data of the laminar error model
for reg in REG:
    INPUT_TURB_FILES[f'aairfoil_2M_full2Dfield_down0p005_up1_laminar_{reg}'] = f'./data/aairfoil_2M_full2Dfield_down0p005_up1_laminar_{reg}.{EXT}'
    TURB_CASES.append(f'aairfoil_2M_full2Dfield_down0p005_up1_laminar_{reg}')
    DATASET_PLOT_TITLE[f'aairfoil_2M_full2Dfield_down0p005_up1_laminar_{reg}'] = f'A-Airfoil, Reynolds number: 2M, Full 2D field, laminar, {reg} region'

# NOTE: Gaussian data (2M)
RE = 2
POS = ['MAPG', 'FPG', 'APG', 'SEP']
INPUT_TURB_FILES[f'gaussian_{RE}M'] = f'./data/gaussian_{RE}M_data_ALL.{EXT}'
TURB_CASES.append(f'gaussian_{RE}M')
DATASET_PLOT_TITLE[f'gaussian_{RE}M'] = f'Gaussian Bump: {RE}M'
for pos in POS:
    INPUT_TURB_FILES[f'gaussian_{RE}M_{pos}'] = f'./data/gaussian_{RE}M_data_{pos}.{EXT}'
    TURB_CASES.append(f'gaussian_{RE}M_{pos}')
    DATASET_PLOT_TITLE[f'gaussian_{RE}M_{pos}'] = f'Gaussian Bump: {RE}M {pos}'

# NOTE: Gaussian data (1M)
RE = 1
POS = ['MAPG', 'FPG_concave', 'FPG_convex', 'APG_stable', 'APG', 'SEP']
INPUT_TURB_FILES[f'gaussian_{RE}M'] = f'./data/gaussian_{RE}M_data_ALL.{EXT}'
TURB_CASES.append(f'gaussian_{RE}M')
DATASET_PLOT_TITLE[f'gaussian_{RE}M'] = f'Gaussian Bump: {RE}M'
for pos in POS:
    INPUT_TURB_FILES[f'gaussian_{RE}M_{pos}'] = f'./data/gaussian_{RE}M_data_{pos}.{EXT}'
    TURB_CASES.append(f'gaussian_{RE}M_{pos}')
    DATASET_PLOT_TITLE[f'gaussian_{RE}M_{pos}'] = f'Gaussian Bump: {RE}M {pos}'

# NOTE: Different bubble cases
BUB_SUB = ['A', 'B', 'C', 'K']
for sub in BUB_SUB:
    INPUT_TURB_FILES[f'bub_{sub}'] = f'./data/bub_{sub}_data.{EXT}'
    TURB_CASES.append(f'bub_{sub}')
    DATASET_PLOT_TITLE[f'bub_{sub}'] = f'Pressure-induced Separation bubble, Subcase: {sub}'
# NOTE: Bubble with swept
sub = 'C35'
INPUT_TURB_FILES[f'bub_{sub}'] = f'./data/bub_{sub}_data.{EXT}'
TURB_CASES.append(f'bub_{sub}')
DATASET_PLOT_TITLE[f'bub_{sub}'] = f'Pressure-induced Separation bubble with swept, Subcase: {sub}'

# NOTE: naca4412 with different Reynolds numbers
NACA_SUB = ['10', '4', '2', '1']
for sub in NACA_SUB:
    INPUT_TURB_FILES[f'naca_4412_{sub}'] = f'./data/naca_4412_top{sub}n_data.{EXT}'
    TURB_CASES.append(f'naca_4412_{sub}')
    DATASET_PLOT_TITLE[f'naca_4412_{sub}'] = f'NACA4412 Re={sub}00000'
# NOTE: naca0012 (turbulent, KTH) 
INPUT_TURB_FILES[f'naca_0012'] = f'./data/naca_0012_top4n12_data.{EXT}'

# NOTE: NACA0025 from Konrad (private communication)
INPUT_TURB_FILES[f'naca_0025'] = f'./data/naca0025_Konrad_data.{EXT}'
DATASET_PLOT_TITLE[f'naca_0025'] = f'NACA0025'
REGION = ['Laminar', 'Transition', 'Turbulent']
for rg in REGION:
    INPUT_TURB_FILES[f'naca_0025_{rg}'] = f'./data/naca0025_{rg}_Konrad_data.{EXT}'
    TURB_CASES.append(f'naca_0025_{rg}')
    DATASET_PLOT_TITLE[f'naca_0025_{rg}'] = f'NACA_0025: {rg} region'

# NOTE: Falkner-Skan laminar boundary layer
LBL_CASES = ['ZPG','FPG','APG', 'ALL','m_a0p08','m_a0p025', 'a0p6']
for case in LBL_CASES:
    INPUT_TURB_FILES[f'FS_{case}'] = f'./data/FS_{case}_data.{EXT}'
    TURB_CASES.append(f'FS_{case}')
    DATASET_PLOT_TITLE[f'FS_{case}'] = f'Laminar Boundary Layer: {case}'

# NOTE: Stokes Second 
INPUT_TURB_FILES[f'Stokes2'] = f'./data/Stokes2_data.{EXT}'
TURB_CASES.append(f'Stokes2')
DATASET_PLOT_TITLE[f'Stokes2'] = f'Stokes Second Problem'

# NOTE: Oscillating pipe
INPUT_TURB_FILES[f'OscPipe'] = f'./data/OscPipe_data.{EXT}'
TURB_CASES.append(f'OscPipe')
DATASET_PLOT_TITLE[f'OscPipe'] = f'Oscillating Pipe'

# NOTE: Transition ZPG boundary layer from JHU
INPUT_TURB_FILES[f'Transition'] = f'./data/transition_JHU_data.{EXT}'
TURB_CASES.append(f'Transition')
DATASET_PLOT_TITLE[f'Transition'] = f'Transition ZPG Boundary Layer'
# NOTE: Transition ZPG boundary layer from JHU
REGION = ['laminar', 'transition', 'turbulent']
for rg in REGION:
    INPUT_TURB_FILES[f'Transition_{rg}'] = f'./data/transition_JHU_{rg}_data.{EXT}'
    TURB_CASES.append(f'Transition_{rg}')
    DATASET_PLOT_TITLE[f'Transition_{rg}'] = f'Transition ZPG Boundary Layer: {rg} region'

# NOTE: Transition JHU full 2D field with wall-normal sampling range specified
INPUT_TURB_FILES[f'transition_JHU_full2Dfield_down0_up1'] = f'./data/transition_JHU_full2Dfield_down0_up1.{EXT}'
TURB_CASES.append(f'transition_JHU_full2Dfield_down0_up1')
DATASET_PLOT_TITLE[f'transition_JHU_full2Dfield_down0_up1'] = f'Transition ZPG Boundary Layer'

for rg in REGION:
    INPUT_TURB_FILES[f'transition_JHU_full2Dfield_down0_up1_{rg}'] = f'./data/transition_JHU_full2Dfield_down0_up1_{rg}.{EXT}'
    TURB_CASES.append(f'transition_JHU_full2Dfield_down0_up1_{rg}')
    DATASET_PLOT_TITLE[f'transition_JHU_full2Dfield_down0_up1_{rg}'] = f'Transition ZPG Boundary Layer: {rg} region'

# NOTE: NACA0012 laminar (Rec=5000)
INPUT_TURB_FILES[  f'naca0012_laminar'] = f'./data/naca0012_laminar_data.{EXT}'
TURB_CASES.append( f'naca0012_laminar')
DATASET_PLOT_TITLE[f'naca0012_laminar'] = f'Laminar NACA0012 (Rec=5000)'

# NOTE: Cases with curvature from Applebaum et al. (2025)
INPUT_TURB_FILES[f'curved_TBL'] = f'./data/curved_TBL_data.{EXT}'
TURB_CASES.append(f'curved_TBL')
DATASET_PLOT_TITLE[f'curved_TBL'] = f'Boundary layer with curvature'
REGION = ['APG', 'FPG']
for r in REGION:
    INPUT_TURB_FILES[f'curved_TBL_{r}'] = f'./data/curved_TBL_{r}_data.{EXT}'
    TURB_CASES.append(f'curved_TBL_{r}')
    DATASET_PLOT_TITLE[f'curved_TBL_{r}'] = f'Boundary layer with curvature: {r}'

# NOTE: TBL by Volino(2020)
CASE = np.arange(1, 9)
for c in CASE:
    INPUT_TURB_FILES[f'TBL_Volino_{c}'] = f'./data/TBL_Volino_{c}_data.{EXT}'
    TURB_CASES.append(f'TBL_Volino_{c}')
    DATASET_PLOT_TITLE[f'TBL_Volino_{c}'] = f'Turbulent Boundary layer with first FPG and APG by Volino (2020), Case: {c}'

# NOTE: TBL by Gungor(2024)
CASE = [16, 22, 23]
for c in CASE:
    INPUT_TURB_FILES[f'TBL_Gungor_{c}'] = f'./data/TBL_Gungor_20{c}_data.{EXT}'
    TURB_CASES.append(f'TBL_Gungor_{c}')
    DATASET_PLOT_TITLE[f'TBL_Gungor_{c}'] = f'Turbulent Boundary layer by Gungor (2024), Case: {c}'

# NOTE: Periodic Hill from Xiao et al. (2020)
PH_CASES = ['X']
for case in PH_CASES:
    INPUT_TURB_FILES[f'ph_{case}'] = f'./data/ph_{case}_data.{EXT}'
    TURB_CASES.append(f'ph_{case}')
    DATASET_PLOT_TITLE[f'ph_{case}'] = f'Periodic Hill: {case}'

# NOTE: ph_G from Gloerfelt et al. (2019)
RE = [2800, 10595, 19000, 37000]
for re in RE:
    INPUT_TURB_FILES[f'ph_G_{re}'] = f'./data/ph_G_{re}_data.{EXT}'
    TURB_CASES.append(f'ph_G_{re}')
    DATASET_PLOT_TITLE[f'ph_G_{re}'] = f'Periodic Hill by Gloerfelt: {case}'

# NOTE: ph_B: Full 2D field with wall-normal sampling range specified
INPUT_TURB_FILES[f'ph_B_full2Dfield_down0_up1'] = f'./data/ph_B_full2Dfield_fixeddelta_delta1_down0_up1.{EXT}'
TURB_CASES.append(f'ph_B_full2Dfield_down0_up1')
DATASET_PLOT_TITLE[f'ph_B_full2Dfield_down0_up1'] = f'Periodic Hill: ph_B'

# NOTE: NASA Hump Uzun et al. (2017) with PG zones
REGION = ['APG', 'FPG', 'APG_SEP1', 'APG_SEP2', 'FPG_SEP']
for r in REGION:
    INPUT_TURB_FILES[f'hump_{r}'] = f'./data/hump_{r}_data.{EXT}'
    TURB_CASES.append(f'hump_{r}')
    DATASET_PLOT_TITLE[f'hump_{r}'] = f'NASA Hump: {r}'

# NOTE: Aposteriori Channel Re
casename = 'aposteriori_CH4200'
GRID_LEVEL = ['C', 'N', 'F']
for case in GRID_LEVEL:
    INPUT_TURB_FILES[f'{casename}_{case}'] = f'./data/{casename}_{case}.{EXT}'
    TURB_CASES.append(f'{casename}_{case}')
    DATASET_PLOT_TITLE[f'{casename}_{case}'] = f'{casename}_{case}'

# NOTE: Aposteriori High Lift CRM
casename = 'aposteriori_CRM'
AOA = ['6', '17', '23']
for case in AOA:
    INPUT_TURB_FILES[f'{casename}_aoa{case}'] = f'./data/{casename}_aoa{case}.{EXT}'
    TURB_CASES.append(f'{casename}_aoa{case}')
    DATASET_PLOT_TITLE[f'{casename}_aoa{case}'] = f'{casename}_aoa{case}'

# NOTE: Aposteriori High Lift CRM (no flaps) new data with x,y,z
casename = 'aposteriori_CRM_tc2p2'
AOA = ['6', '17', '23']
for case in AOA:
    INPUT_TURB_FILES[f'{casename}_aoa{case}_new'] = f'./data/{casename}_aoa{case}_new.{EXT}'
    TURB_CASES.append(f'{casename}_aoa{case}_new')
    DATASET_PLOT_TITLE[f'{casename}_aoa{case}_new'] = f'{casename}_aoa{case}_new'

# NOTE: Aposteriori High Lift CRM (no flaps) new data with x,y,z and with long time WM input filtering
casename = 'aposteriori_CRM_tc2p2'
AOA = ['6', '17', '23']
for case in AOA:
    INPUT_TURB_FILES[f'{casename}_aoa{case}_snapshot100'] = f'./data/{casename}_aoa{case}_snapshot100.{EXT}'
    TURB_CASES.append(f'{casename}_aoa{case}_snapshot100')
    DATASET_PLOT_TITLE[f'{casename}_aoa{case}_snapshot100'] = f'{casename}_aoa{case}_snapshot100'

# NOTE: Aposteriori High Lift CRM (with flaps) new data with x,y,z and with long time WM input filtering
casename = 'aposteriori_CRM_tc2p3'
AOA = ['6', '17', '23']
for case in AOA:
    INPUT_TURB_FILES[f'{casename}_aoa{case}_snapshot100'] = f'./data/{casename}_aoa{case}_snapshot100.{EXT}'
    TURB_CASES.append(f'{casename}_aoa{case}_snapshot100')
    DATASET_PLOT_TITLE[f'{casename}_aoa{case}_snapshot100'] = f'{casename}_aoa{case}_snapshot100'

# NOTE: Aposteriori High Lift CRM (with flaps) new data with x,y,z and with long time WM input filtering + fuselage refined L10
casename = 'aposteriori_CRM_tc2p3'
AOA = ['6', '17', '23']
for case in AOA:
    INPUT_TURB_FILES[f'{casename}_aoa{case}_fuseL10_snapshot100'] = f'./data/{casename}_aoa{case}_fuseL10_snapshot100.{EXT}'
    TURB_CASES.append(f'{casename}_aoa{case}_fuseL10_snapshot100')
    DATASET_PLOT_TITLE[f'{casename}_aoa{case}_fuseL10_snapshot100'] = f'{casename}_aoa{case}_fuseL10_snapshot100'

# NOTE: Aposteriori Gaussian
casename = 'aposteriori_Gaussian'
INPUT_TURB_FILES[f'{casename}'] = f'./data/{casename}.{EXT}'
TURB_CASES.append(f'{casename}')
DATASET_PLOT_TITLE[f'{casename}'] = f'{casename}'

# NOTE: Aposteriori Gaussian testing
casename = 'aposteriori_Gaussian_testing'
INPUT_TURB_FILES[f'{casename}'] = f'./data/{casename}.{EXT}'
TURB_CASES.append(f'{casename}')
DATASET_PLOT_TITLE[f'{casename}'] = f'{casename}'

# NOTE: Aposteriori Gaussian centerline
casename = 'aposteriori_Gaussian_CL'
INPUT_TURB_FILES[f'{casename}'] = f'./data/{casename}.{EXT}'
TURB_CASES.append(f'{casename}')
DATASET_PLOT_TITLE[f'{casename}'] = f'{casename}'

# NOTE: Aposteriori Gaussian centerline averaged
casename = 'aposteriori_Gaussian_CL_tavg'
INPUT_TURB_FILES[f'{casename}'] = f'./data/{casename}.{EXT}'
TURB_CASES.append(f'{casename}')
DATASET_PLOT_TITLE[f'{casename}'] = f'{casename}'

# NOTE: Aposteriori Gaussian centerline signed
casename = 'aposteriori_Gaussian_CL_sign'
INPUT_TURB_FILES[f'{casename}'] = f'./data/{casename}.{EXT}'
TURB_CASES.append(f'{casename}')
DATASET_PLOT_TITLE[f'{casename}'] = f'{casename}'

# NOTE: Aposteriori Gaussian centerline signed averaged
casename = 'aposteriori_Gaussian_CL_sign_tavg'
INPUT_TURB_FILES[f'{casename}'] = f'./data/{casename}.{EXT}'
TURB_CASES.append(f'{casename}')
DATASET_PLOT_TITLE[f'{casename}'] = f'{casename}'

# NOTE: Aposteriori Gaussian signed
casename = 'aposteriori_Gaussian_sign'
INPUT_TURB_FILES[f'{casename}'] = f'./data/{casename}.{EXT}'
TURB_CASES.append(f'{casename}')
DATASET_PLOT_TITLE[f'{casename}'] = f'{casename}'

# NOTE: Aposteriori Gaussian signed combined snapshots
casename = 'aposteriori_Gaussian_combsnaps20_sign'
INPUT_TURB_FILES[f'{casename}'] = f'./data/{casename}.{EXT}'
TURB_CASES.append(f'{casename}')
DATASET_PLOT_TITLE[f'{casename}'] = f'{casename}'

# NOTE: Aposteriori Gaussian CL signed combined snapshots
casename = 'aposteriori_Gaussian_combsnaps100_CL_sign'
INPUT_TURB_FILES[f'{casename}'] = f'./data/{casename}.{EXT}'
TURB_CASES.append(f'{casename}')
DATASET_PLOT_TITLE[f'{casename}'] = f'{casename}'

# NOTE: Aposteriori Gaussian CL signed averaged combined snapshots
casename = 'aposteriori_Gaussian_combsnaps100_CL_sign_tavg'
INPUT_TURB_FILES[f'{casename}'] = f'./data/{casename}.{EXT}'
TURB_CASES.append(f'{casename}')
DATASET_PLOT_TITLE[f'{casename}'] = f'{casename}'

# NOTE: Aposteriori TBL at 5deg
casename = 'aposteriori_TBL_5'
INPUT_TURB_FILES[f'{casename}'] = f'./data/{casename}.{EXT}'
TURB_CASES.append(f'{casename}')
DATASET_PLOT_TITLE[f'{casename}'] = f'{casename}'

# NOTE: Aposteriori TBL at 5deg averaged
casename = 'aposteriori_TBL_5_averaged'
INPUT_TURB_FILES[f'{casename}'] = f'./data/{casename}.{EXT}'
TURB_CASES.append(f'{casename}')
DATASET_PLOT_TITLE[f'{casename}'] = f'{casename}'

# NOTE: Aposteriori TBL at 5deg coarse
casename = 'aposteriori_TBL_5_coarse'
INPUT_TURB_FILES[f'{casename}'] = f'./data/{casename}.{EXT}'
TURB_CASES.append(f'{casename}')
DATASET_PLOT_TITLE[f'{casename}'] = f'{casename}'

# NOTE: Aposteriori TBL at 5deg medium
casename = 'aposteriori_TBL_5_medium'
INPUT_TURB_FILES[f'{casename}'] = f'./data/{casename}.{EXT}'
TURB_CASES.append(f'{casename}')
DATASET_PLOT_TITLE[f'{casename}'] = f'{casename}'

# NOTE: Aposteriori TBL at 5deg filtered dpdx
casename = 'aposteriori_TBL_5_filtered_dpdx'
INPUT_TURB_FILES[f'{casename}'] = f'./data/{casename}.{EXT}'
TURB_CASES.append(f'{casename}')
DATASET_PLOT_TITLE[f'{casename}'] = f'{casename}'

# NOTE: Aposteriori TBL at 5deg filtered dpdx long
casename = 'aposteriori_TBL_5_filtered_dpdx_long'
INPUT_TURB_FILES[f'{casename}'] = f'./data/{casename}.{EXT}'
TURB_CASES.append(f'{casename}')
DATASET_PLOT_TITLE[f'{casename}'] = f'{casename}'

def print_dataset_tree(datasets, indent=0):
    for key, value in datasets.items():
        print('    ' * indent + f"├── {key}")
        if isinstance(value, dict) and value:  # Only recurse if the value is a non-empty dictionary
            print_dataset_tree(value, indent + 1)

            # Dataset dictionary representing a tree-like structure
TURB_CASES_TREE = {
    "CH": {},
    "SYN": {},
    "PIPE": {},
    "apg": {
        "apg_b1n": {},
        "apg_b2n": {},
        "apg_m13n": {},
        "apg_m16n": {},
        "apg_m18n": {}
    },
    "bub": {
        "bub_A": {},
        "bub_B": {},
        "bub_C": {},
        "bub_K": {}
    },
    "TBL": {
        "TBL_-4": {},
        "TBL_-3": {},
        "TBL_-2": {},
        "TBL_-1": {},
        "TBL_5": {},
        "TBL_10": {},
        "TBL_15": {},
        "TBL_20": {}
    },
    "naca_0012": {},
    "naca_0025": {},
    "naca_4412": {
        "naca_4412_10": {},
        "naca_4412_4": {},
        "naca_4412_2": {},
        "naca_4412_1": {}
    },
    "aairfoilapg_kth": {},
    "gaussian_2M": {},
    "curve": {},
    "aairfoil_2M": {},
    "aairfoil_10M": {},
    "gaussian_1M": {
        "gaussian_1M_MAPG": {},
        "gaussian_1M_FPG": {},
        "gaussian_1M_APG": {},
        "gaussian_1M_SEP": {},
        "gaussian_1M_concave": {},
        "gaussian_1M_convex": {},
        "gaussian_1M_FPG_concave": {},
        "gaussian_1M_FPG_convex": {}
    },
    "gaussian_2M": {
        "gaussian_2M_MAPG": {},
        "gaussian_2M_FPG": {},
        "gaussian_2M_APG": {},
        "gaussian_2M_SEP": {},
        "gaussian_2M_concave": {},
        "gaussian_2M_convex": {},
        "gaussian_2M_FPG_concave": {},
        "gaussian_2M_FPG_convex": {}
    },
    "FS_ZPG": {},
    "FS_FPG": {},
    "FS_APG": {},
    "curve_pg": {},
    "bend": {},
    "convdiv": {},
    "backstep": {},
    "ph_B": {},
}
###############################
# NOTE: Go over all cases and add a "stencil" version 
#       Currently, we only support a few cases
#       1. TBL
#       2. apg by KTH
#       3. Channel and Pipe
TURB_CASES_WITH_STENCIL = ['TBL_5', 'CH', 'PIPE', 'apg_b1n', 'apg_b2n', 'apg_m13n', 'apg_m16n', 'apg_m18n', 'TBL_-4', 'TBL_-3', 'TBL_-2', 'TBL_-1', 'TBL_5', 'TBL_10', 'TBL_15', 'TBL_20']
for case in TURB_CASES_WITH_STENCIL:
    TURB_CASES += [f'{case}_stencil']
    INPUT_TURB_FILES[f'{case}_stencil'] = f'./data/stencil/{case}_data_stencils.{EXT}'

#################
# NOTE: Here we also define whether for each cases, we have extra cases focusing on certain cases
#
STATION = {
        "bend": [-0.193, 0.02499936, 0.17799969, 0.3300006 , 0.48299949, 0.9400002 , 1.245], # The station right after 30 degree bend
        "convdiv": [0, 2.0203, 4.0488, 5.07059, 5.27287, 5.50133, 5.60819, 6.63815, 7.12181, 7.46414, 8.05638],
        "backstep": [-4.,  1.,  4.,  6., 10.],
        "ph_B": [1.1475, 4.9719, 5.827861, 6.052468, 6.373975, 6.775461, 8.002593, 8.354093],
        "gaussian_2M_MAPG": [-0.59959855, -0.40889207],
        "gaussian_2M_FPG": [-0.2, -0.15, -0.1, ],
        "gaussian_2M_APG": [2.84180234e-05, 4.99274804e-02],
        "gaussian_2M_SEP": [0.2, 0.3597485],
        "gaussian_1M_MAPG": [-0.59999815,-0.40040712],
        "gaussian_1M_FPG_concave": [-0.22034062, -0.16039],
        "gaussian_1M_FPG_convex": [-0.10079176,  -0.04998937, -0.00222268],
        "gaussian_1M_APG_stable": [0.00998682,0.02995354,0.04391049],
        "gaussian_1M_APG": [0.08037356, 0.10057804,0.17514278],
        "hump": [-0.8, -0.4, 0, 0.10116047, 0.2014175, 0.30294264, 0.50484454, 0.64700406, 0.66666904, 0.77782227, 0.78822388, 0.79862236, 0.90906449, 1.00000186, 1.2, 1.4],
        "bub_A": [-5.02708292, -2.9958334, -1.1677103,0.0510397,0.25417042,0.3015604],
        "bub_B": [-5.1187501, -2.07188034,-1.1984396,0.32499981,1.99061966,3.00625038],
        "aairfoil_10M": [0.05 , 0.08 , 0.1  , 0.15 , 0.2  , 0.5  , 0.825, 0.93, 0.99],
        "aairfoil_2M": [0.030006787, 0.08063563, 0.10109831 , 0.15148925, 0.30277907, 0.5055756 ,0.82811433, 0.93907689, 0.94093326],
        "smoothramp": [-0.72, -0.2, 0, 0.3, 1.5, 2, 2.5, 3]
}

# NOTE: Add subcases for each station
for case in STATION:
    for i, station in enumerate(STATION[case]):
        # TURB_CASES_TREE[case][f'station_{i}'] = {}
        TURB_CASES.append(f'{case}_station_{i}')
        DATASET_PLOT_TITLE[f'{case}_station_{i}'] = f'{DATASET_PLOT_TITLE[case]}, Station: {station}'

