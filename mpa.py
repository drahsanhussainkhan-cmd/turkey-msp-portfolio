from qgis.core import *
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtCore import QRectF

# Get project and layers
project = QgsProject.instance()
layers = {layer.name(): layer for layer in project.mapLayers().values()}

# --- Style Land Layer ---
land = layers['ne_10m_land.shp']
land_symbol = QgsFillSymbol.createSimple({
    'color': '#F5F0E8',
    'outline_color': '#AAAAAA',
    'outline_width': '0.3'
})
land.renderer().setSymbol(land_symbol)
land.triggerRepaint()

# --- Style EEZ Layer ---
eez = layers['World_EEZ_v12_20231025/eez_v12.shp']
eez_symbol = QgsFillSymbol.createSimple({
    'color': '173,216,230,80',
    'outline_color': '#2166AC',
    'outline_width': '0.8',
    'outline_style': 'dash'
})
eez.renderer().setSymbol(eez_symbol)
eez.triggerRepaint()

# --- Style MPA Polygons ---
mpa = layers['WDPA_WDOECM_Feb2026_Public_TUR_shp-polygons']
mpa_symbol = QgsFillSymbol.createSimple({
    'color': '0,128,0,150',
    'outline_color': '#005500',
    'outline_width': '0.6'
})
mpa.renderer().setSymbol(mpa_symbol)
mpa.triggerRepaint()

# --- Filter EEZ to Turkey only ---
eez_layer = layers['World_EEZ_v12_20231025/eez_v12.shp']
eez_layer.setSubsetString("\"SOVEREIGN1\" = 'Turkey'")

# --- Set ocean background color ---
iface.mapCanvas().setCanvasColor(QColor('#AED9E0'))
iface.mapCanvas().refresh()

# --- Create Print Layout ---
manager = project.layoutManager()
existing = manager.layoutByName("Turkey MPA Map")
if existing:
    manager.removeLayout(existing)

layout = QgsPrintLayout(project)
layout.initializeDefaults()
layout.setName("Turkey MPA Map")
manager.addLayout(layout)

# Page size A3 landscape
pc = layout.pageCollection()
pc.pages()[0].setPageSize(QgsLayoutSize(420, 297))

# Map item
map_item = QgsLayoutItemMap(layout)
layout.addLayoutItem(map_item)
map_item.attemptMove(QgsLayoutPoint(5, 20))
map_item.attemptResize(QgsLayoutSize(290, 270))
map_item.setExtent(QgsRectangle(25.0, 35.0, 45.0, 43.0))
map_item.refresh()

# Title
title = QgsLayoutItemLabel(layout)
layout.addLayoutItem(title)
title.setText("Marine Protected Areas of Turkey")
title.setFont(QFont("Arial", 16, QFont.Bold))
title.setFontColor(QColor("#1a1a2e"))
title.attemptMove(QgsLayoutPoint(5, 5))
title.attemptResize(QgsLayoutSize(290, 14))

# Scale bar
scale_bar = QgsLayoutItemScaleBar(layout)
layout.addLayoutItem(scale_bar)
scale_bar.setLinkedMap(map_item)
scale_bar.applyDefaultSettings()
scale_bar.setStyle("Single Box")
scale_bar.setNumberOfSegments(4)
scale_bar.setUnits(QgsUnitTypes.DistanceKilometers)
scale_bar.attemptMove(QgsLayoutPoint(10, 275))

# Legend
legend = QgsLayoutItemLegend(layout)
layout.addLayoutItem(legend)
legend.setLinkedMap(map_item)
legend.setTitle("Legend")
legend.attemptMove(QgsLayoutPoint(300, 20))
legend.attemptResize(QgsLayoutSize(115, 100))

# Export map
exporter = QgsLayoutExporter(layout)
export_path = r"C:\Users\ahk79\OneDrive\Desktop\project 1 msp\turkey_mpa_map.png"
settings = QgsLayoutExporter.ImageExportSettings()
settings.dpi = 300
result = exporter.exportToImage(export_path, settings)

if result == QgsLayoutExporter.Success:
    print("Map exported successfully!")
else:
    print(f"Export failed with code: {result}")