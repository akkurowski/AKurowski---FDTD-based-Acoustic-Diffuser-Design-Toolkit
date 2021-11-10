from itertools import product
from _imports import *
import pandas as pd

# def get_air_speed(p_air_hPa, rho_air):
# def get_air_density(T_air_C, p_air_hPa, RH):
# def get_air_properties(T_air_C=25, p_air_hPa=1013.25, RH=0.5):

T_air_C = [-20,0,20]
p_air_hPa = [990,1020,1050]
RH = [0.3,0.5,0.7]

output_dict = {}
output_dict.update({'T_air_C':[]})
output_dict.update({'p_air_hPa':[]})
output_dict.update({'RH':[]})
output_dict.update({'c':[]})
output_dict.update({'rho':[]})
for p in product(T_air_C,p_air_hPa,RH):
    c, rho = get_air_properties(*p)
    print(p, c, rho)
    output_dict['T_air_C'].append(p[0])
    output_dict['p_air_hPa'].append(p[1])
    output_dict['RH'].append(p[2])
    output_dict['c'].append(c)
    output_dict['rho'].append(rho)

output_dict = pd.DataFrame(output_dict)
output_dict.to_excel('air_props_demo.xlsx')