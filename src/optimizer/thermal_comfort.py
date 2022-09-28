from pythermalcomfort.models import pmv_ppd, clo_tout
from pythermalcomfort.utilities import v_relative, clo_dynamic
from pythermalcomfort.utilities import met_typical_tasks

# input variables
tdb = 27  # dry bulb air temperature, [Celsius]
tr = 25  # mean radiant temperature, [Celsius]
v = 0.3  # average air speed, [m/s]
rh = 50  # relative humidity, [%]
tout = 20 #outdoor temperature at 6 a.m. (to predict clothes) [Celsius]

activity = "Seated, quiet"
met = met_typical_tasks[activity]  #metabolic rate associated to the activity

vr = v_relative(v=v, met=met) #relative air velocity considering the activity
clo = clo_tout(tout) #clothing level given outdoor temp

results = pmv_ppd(tdb=tdb, tr=tr, vr=vr, rh=rh, met=met, clo=clo, standard="ISO")
print(results)

ppd = results['ppd']

if ppd < 6:
    print("Category I")
elif ppd < 10:
    print('Category II')
elif ppd < 15:
    print('Category III')
else:
    print("Does not comply with EN-16798")       