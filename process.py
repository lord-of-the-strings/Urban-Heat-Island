#dependency of plot.py
import numpy as np
import rasterio
print('[*] Loading raw datasets.........')
with rasterio.open('data/ahmedabad_B4.tif') as r_band:
    red=r_band.read(1).astype('float32')
    meta=r_band.meta
with rasterio.open('data/ahmedabad_B5.tif') as nir_band:
    nir = nir_band.read(1).astype('float32')
with rasterio.open('data/ahmedabad_B10.tif') as thermal_band:
    thermal = thermal_band.read(1).astype('float32')
print("[*] Processing metrics........")
with np.errstate(divide='ignore',invalid='ignore'):
    ndvi=(nir-red)/(nir+red)
    ndvi=np.nan_to_num(ndvi,nan=0.0)
kelvin=(thermal*0.00341802)+149.0
celsius=kelvin-273.15
celsius[celsius<0]=np.nan
celsius=np.nan_to_num(celsius,nan=np.nanmean(celsius))
print(f"[+] NDVI Calculated (Min: {ndvi.min():.2f}, Max: {ndvi.max():.2f})")
print(f"[+] LST Calculated in Celsius (Mean Urban Temp: {np.mean(celsius):.1f}°C)")
meta.update(dtype=rasterio.float32, count=1)
with rasterio.open('data/ahmedabad_NDVI.tif', 'w', **meta) as dst:
    dst.write(ndvi, 1)
with rasterio.open('data/ahmedabad_LST.tif', 'w', **meta) as dst:
    dst.write(celsius.astype('float32'), 1)
print("\n[+] Success! 'ahmedabad_NDVI.tif' and 'ahmedabad_LST.tif' generated.")