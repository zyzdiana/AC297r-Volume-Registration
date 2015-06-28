import numpy as np
from Bessel_Interp import bessel_halton_cost_func
from Bessel_Interp import bessel_halton_in_circ_cost_func
from Bessel_Interp import hann_windowed_bessel_halton_in_circ_cost_func
from get_data import get_volume_1
from mask import hann_3D_window


testVol1 = get_volume_1("test_data/vNav_Initial/EPINavigators_Rep_1.dat", 32);
testVol2 = get_volume_1("test_data/vNav_5_deg_RL/EPINavigators_Rep_1.dat", 32);

#print bessel_halton_cost_func(testVol1, testVol2, 256, np.arange(-10,10,0.5), 0)
#print bessel_halton_in_circ_cost_func(testVol1, testVol2, 11, 256, np.arange(-10,10,0.5), 0)
#print hann_windowed_bessel_halton_in_circ_cost_func(testVol1, testVol2, 11, 256, np.arange(-10,10,0.5), 0)
#print bessel_halton_cost_func(testVol1, testVol2, 256, np.array([0]), 0)

testVol1Masked = hann_3D_window(testVol1, 16)
testVol2Masked = hann_3D_window(testVol2, 16)

#print (bessel_halton_cost_func(testVol1Masked, testVol2Masked, 256, np.arange(-10,10,0.5), 0) * 100.0)
#print (bessel_halton_in_circ_cost_func(testVol1Masked, testVol2Masked, 11, 256, np.arange(-10,10,0.5), 0) * 100.0)
#print (hann_windowed_bessel_halton_in_circ_cost_func(testVol1Masked, testVol2Masked, 11, 256, np.arange(-10,10,0.5), 0) * 100.0)


print (hann_windowed_bessel_halton_in_circ_cost_func(testVol1Masked, testVol2Masked, 11, 16, 4, np.arange(-5.5,-4.5,0.01), 0) * 100.0)
print (hann_windowed_bessel_halton_in_circ_cost_func(testVol1Masked, testVol2Masked, 11, 256, 4, np.arange(-5.5,-4.5,0.01), 0) * 100.0)
print (hann_windowed_bessel_halton_in_circ_cost_func(testVol1Masked, testVol2Masked, 11, 1024, 4, np.arange(-5.5,-4.5,0.01), 0) * 100.0)

