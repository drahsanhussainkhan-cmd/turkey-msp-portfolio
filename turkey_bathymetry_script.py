from qgis.core import *
from PyQt5.QtGui import QColor, QFont

project = QgsProject.instance()
layers = {layer.name(): layer for layer in project.mapLayers().values()}

# --- Style GEBCO Bathymetry ---
gebco = layers['Ocean Depth']
shader = QgsColorRampShader()
shader.setColorRampType(QgsColorRampShader.Interpolated)
color_list = [
    QgsColorRampShader.ColorRampItem(-4500, QColor('#08306b'), 'Deep (-4500m)'),
    QgsColorRampShader.ColorRampItem(-3000, QColor('#2171b5'), '-3000m'),
    QgsColorRampShader.ColorRampItem(-2000, QColor('#4292c6'), '-2000m'),
    QgsColorRampShader.ColorRampItem(-1000, QColor('#6baed6'), '-1000m'),
    QgsColorRampShader.ColorRampItem(-500,  QColor('#9ecae1'), '-500m'),
    QgsColorRampShader.ColorRampItem(-200,  QColor('#c6dbef'), '-200m'),
    QgsColorRampShader.ColorRampItem(-50,   QColor('#deebf7'), 'Shallow (-50m)'),
    QgsColorRampShader.ColorRampItem(0,     QColor('#F5F0E8'), 'Land (0m)'),
]
shader.setColorRampItemList(color_list)
raster_shader = QgsRasterShader()
raster_shader.setRasterShaderFunction(shader)
renderer = QgsSingleBandPseudoColorRenderer(gebco.dataProvider(), 1, raster_shader)
gebco.setRenderer(renderer)
gebco.triggerRepaint()
print("Bathymetry styled!")

# --- Style EEZ ---
eez = layers['Turkey EEZ Boundary']
eez.setSubsetString("\"SOVEREIGN1\" = 'Turkey'")
props = {
    'color': '0,0,0,0',
    'outline_color': '#ffffff',
    'outline_width': '1.0',
    'outline_style': 'dash'
}
symbol = QgsFillSymbol.createSimple(props)
eez.renderer().setSymbol(symbol)
eez.triggerRepaint()
print("EEZ styled!")

# --- Zoom to Turkey ---
iface.mapCanvas().setExtent(QgsRectangle(24.0, 34.0, 45.0, 44.0))
iface.mapCanvas().refresh()

# --- Create Layout ---
manager = project.layoutManager()
existing = manager.layoutByName("Turkey Bathymetry Map")
if existing:
    manager.removeLayout(existing)

layout = QgsPrintLayout(project)
layout.initializeDefaults()
layout.setName("Turkey Bathymetry Map")
manager.addLayout(layout)

pc = layout.pageCollection()
pc.pages()[0].setPageSize(QgsLayoutSize(420, 297))

map_item = QgsLayoutItemMap(layout)
layout.addLayoutItem(map_item)
map_item.attemptMove(QgsLayoutPoint(5, 20))
map_item.attemptResize(QgsLayoutSize(290, 270))
map_item.setExtent(QgsRectangle(24.0, 34.0, 45.0, 44.0))
map_item.refresh()

title = QgsLayoutItemLabel(layout)
layout.addLayoutItem(title)
title.setText("Bathymetry of Turkish Waters")
title.setFont(QFont("Arial", 16, QFont.Bold))
title.setFontColor(QColor("#1a1a2e"))
title.attemptMove(QgsLayoutPoint(5, 5))
title.attemptResize(QgsLayoutSize(290, 14))

scale_bar = QgsLayoutItemScaleBar(layout)
layout.addLayoutItem(scale_bar)
scale_bar.setLinkedMap(map_item)
scale_bar.applyDefaultSettings()
scale_bar.setStyle("Single Box")
scale_bar.setNumberOfSegments(4)
scale_bar.setUnits(QgsUnitTypes.DistanceKilometers)
scale_bar.attemptMove(QgsLayoutPoint(10, 275))

legend = QgsLayoutItemLegend(layout)
layout.addLayoutItem(legend)
legend.setLinkedMap(map_item)
legend.setTitle("Depth (m)")
legend.attemptMove(QgsLayoutPoint(300, 20))
legend.attemptResize(QgsLayoutSize(115, 150))

# --- Export ---
exporter = QgsLayoutExporter(layout)
export_path = r"C:\Users\ahk79\OneDrive\Desktop\project 1 msp\project 2 msp\turkey_bathymetry_map.png"
settings = QgsLayoutExporter.ImageExportSettings()
settings.dpi = 300
result = exporter.exportToImage(export_path, settings)

if result == QgsLayoutExporter.Success:
    print("Map exported successfully!")
else:
    print(f"Export failed: {result}")