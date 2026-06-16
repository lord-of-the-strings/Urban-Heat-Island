import geemap
Map = geemap.Map()
Map.add_raster('data/ahmedabad_LST.tif', cmap='jet', layer_name='Land Surface Temp')
Map.to_html('ahmedabad_interactive_map.html')