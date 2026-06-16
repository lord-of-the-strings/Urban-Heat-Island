import ee
import geemap
import os

def mask_clouds(image):
    qa = image.select('QA_PIXEL')
    cloud_shadow_mask = 1 << 3
    dilated_cloud_mask = 1 << 4
    cloud_mask = 1 << 5
    is_clear_shadow = qa.bitwiseAnd(cloud_shadow_mask).eq(0)
    is_clear_dilated = qa.bitwiseAnd(dilated_cloud_mask).eq(0)
    is_clear_cloud = qa.bitwiseAnd(cloud_mask).eq(0)
    clean_mask = is_clear_shadow.And(is_clear_dilated).And(is_clear_cloud)
    return image.updateMask(clean_mask)
try:
    print("[*] Initializing project ................")
    ee.Initialize(project='urban-heat-island-499410')
    print("[+] Earth Engine successfully initialized!")
except Exception as e:
    print(f"[x] Initialization failed: {e}")
ahmedabad=ee.Geometry.Point([72.5714, 23.0225])
roi=ahmedabad.buffer(15000).bounds()
landsat=(ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(roi).filterDate('2025-04-15', '2025-06-15').map(mask_clouds).sort('CLOUD_COVER'))
cleanest=landsat.first()
info=cleanest.getInfo()
selected_bands=cleanest.select(['SR_B4', 'SR_B5', 'ST_B10'])
output_dir = 'data'
os.makedirs(output_dir, exist_ok=True)
print("[*] Downloading cropped satellite bands over Ahmedabad..........")
geemap.ee_export_image(
    selected_bands.select('SR_B4'),
    filename=os.path.join(output_dir, 'ahmedabad_B4.tif'),
    scale=30,
    region=roi
)

geemap.ee_export_image(
    selected_bands.select('SR_B5'),
    filename=os.path.join(output_dir, 'ahmedabad_B5.tif'),
    scale=30,
    region=roi
)

geemap.ee_export_image(
    selected_bands.select('ST_B10'),
    filename=os.path.join(output_dir, 'ahmedabad_B10.tif'),
    scale=30,
    region=roi
)

print("[+] Data Acquisition Complete!")
