asym=RPMD_thermal_R2_mildly_anharmonic_a_-0.605_b_0.427_T_0.125_nbeads_32_dt_0.002_asym.txt
sym=RPMD_thermal_R2_mildly_anharmonic_a_-0.605_b_0.427_T_0.125_nbeads_32_dt_0.002_sym.txt
python ../../../../scripts/FFT_2D.py -file1 ${sym} --file2  ${asym} -n1 401 -n2 401 -beta 8.0 -dt 0.1 -lmax 20
#  python ../../../../scripts/vijay.py -file1 ${sym} --file2  ${asym} -n1 401 -n2 401 -beta 8.0 -dt 0.1 
