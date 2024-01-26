import dash
from dash import dcc, dash_table, html, Input, Output
from dash.dependencies import State
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import rasterio
import plotly.express as px
import plotly.subplots as sp
import json
import geopandas as gpd
from datetime import datetime
from scipy.ndimage import gaussian_filter


# Función para cargar y procesar los datos de CSV
def cargar_y_procesar_datos(csv_path, archivo):
    dfAtlanPP = pd.read_csv(f"{csv_path}{archivo}")
    dfAtlanPP.loc['Promedio'] = dfAtlanPP.iloc[1:, 1:].mean(axis=0)
    dfAtlanPP.loc['Promedio', 'Longitud'] = 'Promedio:Lat-Lon'
    return dfAtlanPP

# Rutas relativas para los archivos (ajusta estas rutas según sea necesario)
base_path = "BrianGallardo/WRF-HN-ClimaScope/15-11-2023 12_00/"
csv_path = base_path + "Series de Tiempo/"
raster_path = base_path + "Rasters/"

# Carga los archivos CSV para las series de tiempo
# Carga y procesamiento de los archivos CSV de cada Departamento.
df_Atlantida = cargar_y_procesar_datos(csv_path, 'Departamentos/Atlántida.csv')
df_Colon = cargar_y_procesar_datos(csv_path, 'Departamentos/Colón.csv')
df_Comayagua = cargar_y_procesar_datos(csv_path, 'Departamentos/Comayagua.csv')
df_Copan = cargar_y_procesar_datos(csv_path, 'Departamentos/Copán.csv')
df_Cortes = cargar_y_procesar_datos(csv_path, 'Departamentos/Cortés.csv')
df_Choluteca = cargar_y_procesar_datos(csv_path, 'Departamentos/Choluteca.csv')
df_ElParaiso = cargar_y_procesar_datos(csv_path, 'Departamentos/El Paraíso.csv')
df_FranciscoMorazan = cargar_y_procesar_datos(csv_path, 'Departamentos/Francisco Morazán.csv')
df_GraciasaDios = cargar_y_procesar_datos(csv_path, 'Departamentos/Gracias a Dios.csv')
df_Intibuca = cargar_y_procesar_datos(csv_path, 'Departamentos/Intibucá.csv')
df_Islas = cargar_y_procesar_datos(csv_path, 'Departamentos/Islas de la Bahía.csv')
df_LaPaz = cargar_y_procesar_datos(csv_path, 'Departamentos/La Paz.csv')
df_Lempira = cargar_y_procesar_datos(csv_path, 'Departamentos/Lempira.csv')
df_Ocotepeque = cargar_y_procesar_datos(csv_path, 'Departamentos/Ocotepeque.csv')
df_Olancho = cargar_y_procesar_datos(csv_path, 'Departamentos/Olancho.csv')
df_SantaBarbara = cargar_y_procesar_datos(csv_path, 'Departamentos/Santa Bárbara.csv')
df_Valle = cargar_y_procesar_datos(csv_path, 'Departamentos/Valle.csv')
df_Yoro = cargar_y_procesar_datos(csv_path, 'Departamentos/Yoro.csv')

# Carga el archivo CSV
dfAtlanPP = pd.read_csv(csv_path + 'Departamentos/Atlántida.csv')

# Establece el índice para que coincida con la imagen
dfAtlanPP.set_index('Longitud', inplace=True)


# Grafica de precipitación
figAtlanPP = go.Figure()

for row in dfAtlanPP.transpose().itertuples():
    figAtlanPP.add_trace(go.Bar(x=dfAtlanPP.index[1:], y=row[2:], name='Lat, Long'))

figAtlanPP.update_layout(
    height=600,
    width=1320,
    template='plotly_white',
    title='Atlántida PP (mm)',
    xaxis_title="Fechas del Pronóstico WRF",
    yaxis_title="Precipitación (mm)"
)
##.........Grafica de promedios
fechas = dfAtlanPP.columns[1]  # Asume que la segunda columna es la fecha
valores = dfAtlanPP.columns[1:]  # Las columnas de datos de precipitación

# Calcular promedio de precipitación para cada columna
promedios = dfAtlanPP[valores].mean(axis=0)  # axis=0 calcula el promedio a lo largo de las columnas

figAavgT = go.Figure()

figAavgT.add_trace(go.Bar(x=dfAtlanPP.index[1:], y=row[2:], name=' PPD (mm)'))
figAavgT.add_trace(go.Scatter(x=dfAtlanPP.index[1:], y=row[2:], name='Tendencia', mode='lines+markers', marker=dict(symbol="diamond", size=10), line=dict(color='cyan')))

figAavgT.update_layout(
    height=500,
    width=1320,
    template='plotly_dark',
    title='Atlántida Precipitación Promedio Diaria (PPD)',
    xaxis_title="Fechas del Pronóstico WRF",
    yaxis_title="Precipitación Promedio Diaria (mm)"
)

# ...

# ...

# Carga el archivo raster para cada Departamento
raster_file_Atlantidaavg = 'BrianGallardo/WRF-HN-ClimaScope/15-11-2023 12_00/Rasters/Departamentos/Atlántida/avg_dia5.tif'
with rasterio.open(raster_file_Atlantidaavg) as src:
    raster_array_Atlantidaavg = src.read(1)
    # Obtener transformación de Rasterio para convertir coordenadas de píxeles a geográficas
    transform = src.transform

    # Calcular las coordenadas geográficas para cada píxel
    height, width = raster_array_Atlantidaavg.shape
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')

# Crear un DataFrame para almacenar las coordenadas y los valores
dfAtlanPPavgRaster = pd.DataFrame({
    'x': np.array(xs).flatten(),
    'y': np.array(ys).flatten(),
    'value': raster_array_Atlantidaavg.flatten()
})

# Cargar el archivo GeoJSON
geojson_pathAPPoligono = 'BrianGallardo/WRF-HN-ClimaScope/15-11-2023 12_00/GeoJson/Departamentos/Atlantida.geojson'
APPoligono_gdf = gpd.read_file(geojson_pathAPPoligono) #RAPavg_ (Raster Atlantida Precipitacio Promedio)

# Extraer las coordenadas del contorno
# Extraer las coordenadas para el contorno
x_geojsonPavg, y_geojsonPavg = [], []
for geom in APPoligono_gdf.geometry:
    if geom.geom_type == 'Polygon':
        x, y = geom.exterior.coords.xy
        x_geojsonPavg.extend(x)
        y_geojsonPavg.extend(y)
    elif geom.geom_type == 'MultiPolygon':
        for part in geom.geoms:  # Iterar sobre cada polígono en el MultiPolygon
            x, y = part.exterior.coords.xy
            x_geojsonPavg.extend(x)
            y_geojsonPavg.extend(y)

# Crear un gráfico con los ejes georreferenciados
fig_map_Atlantidaavg = px.imshow(
    dfAtlanPPavgRaster.pivot(index='y', columns='x', values='value'),
    color_continuous_scale='Blues',
    labels={'color': 'Precipitación (mm)'},
    origin='lower'  # Esto asegura que el gráfico se alinee con las coordenadas geográficas
)

# Añadir la traza del contorno
fig_map_Atlantidaavg.add_trace(go.Scatter(x=x_geojsonPavg, y=y_geojsonPavg, mode='lines', name='Atlantida',
                         line=dict(color='black', width=1)))

fig_map_Atlantidaavg.update_layout(
    title_text='Precipitación Promedio en Atlántida del Pronóstico WRF',
    title_x=0.5,
    title_font=dict(size=14),
    height=500,
    width=1320,
    margin=dict(t=40, b=0, l=0, r=0),
    coloraxis_colorbar=dict(
        title='PP (mm)',
        titleside='right',
        thicknessmode='pixels', thickness=15,
        lenmode='pixels', len=300,
        yanchor='middle', y=0.5,
        titlefont=dict(size=12)
    ),
    xaxis_showgrid=False,  # Oculta la cuadrícula del eje X
    yaxis_showgrid=False   # Oculta la cuadrícula del eje Y
)



#############################################################################
raster_file_Atlantidaac = 'BrianGallardo/WRF-HN-ClimaScope/15-11-2023 12_00/Rasters/Departamentos/Atlántida/ppAcum5dias.tif'
with rasterio.open(raster_file_Atlantidaac) as src:
    raster_array_Atlantidaac = src.read(1)
    # Obtener transformación de Rasterio para convertir coordenadas de píxeles a geográficas
    transform = src.transform

    # Calcular las coordenadas geográficas para cada píxel
    height, width = raster_array_Atlantidaac.shape
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')

# Crear un DataFrame para almacenar las coordenadas y los valores
dfAtlanPPacRaster = pd.DataFrame({
    'x': np.array(xs).flatten(),
    'y': np.array(ys).flatten(),
    'value': raster_array_Atlantidaac.flatten()
})

# Cargar el archivo GeoJSON
geojson_pathAPoligono = 'BrianGallardo/WRF-HN-ClimaScope/15-11-2023 12_00/GeoJson/Departamentos/Atlantida.geojson'
APoligono_gdf = gpd.read_file(geojson_pathAPoligono) #RAPavg_ (Raster Atlantida Precipitacio Promedio)

# Extraer las coordenadas del contorno
# Extraer las coordenadas para el contorno
x_geojsonPac, y_geojsonPac = [], []
for geom in APoligono_gdf.geometry:
    if geom.geom_type == 'Polygon':
        x, y = geom.exterior.coords.xy
        x_geojsonPac.extend(x)
        y_geojsonPac.extend(y)
    elif geom.geom_type == 'MultiPolygon':
        for part in geom.geoms:  # Iterar sobre cada polígono en el MultiPolygon
            x, y = part.exterior.coords.xy
            x_geojsonPac.extend(x)
            y_geojsonPac.extend(y)

# Crear un gráfico con los ejes georreferenciados
fig_map_Atlantidaac = px.imshow(
    dfAtlanPPacRaster.pivot(index='y', columns='x', values='value'),
    color_continuous_scale='Blues',
    labels={'color': 'Precipitación (mm)'},
    origin='lower'  # Esto asegura que el gráfico se alinee con las coordenadas geográficas
)

# Añadir la traza del contorno
fig_map_Atlantidaac.add_trace(go.Scatter(x=x_geojsonPac, y=y_geojsonPac, mode='lines', name='Atlantida',
                         line=dict(color='black', width=1)))

fig_map_Atlantidaac.update_layout(
    title_text='Precipitación Acumulada en Atlántida del Pronóstico WRF',
    title_x=0.5,
    title_font=dict(size=14),
    height=500,
    width=1320,
    margin=dict(t=40, b=0, l=0, r=0),
    coloraxis_colorbar=dict(
        title='PP (mm)',
        titleside='right',
        thicknessmode='pixels', thickness=15,
        lenmode='pixels', len=300,
        yanchor='middle', y=0.5,
        titlefont=dict(size=12)
    ),
    xaxis_showgrid=False,  # Oculta la cuadrícula del eje X
    yaxis_showgrid=False   # Oculta la cuadrícula del eje Y
)
#############################################################################
raster_file_Atlantidamin = 'BrianGallardo/WRF-HN-ClimaScope/15-11-2023 12_00/Rasters/Departamentos/Atlántida/min_dia5.tif'
with rasterio.open(raster_file_Atlantidamin) as src:
    raster_array_Atlantidamin = src.read(1)
    # Obtener transformación de Rasterio para convertir coordenadas de píxeles a geográficas
    transform = src.transform

    # Calcular las coordenadas geográficas para cada píxel
    height, width = raster_array_Atlantidamin.shape
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')

# Crear un DataFrame para almacenar las coordenadas y los valores
dfAtlanPPminRaster = pd.DataFrame({
    'x': np.array(xs).flatten(),
    'y': np.array(ys).flatten(),
    'value': raster_array_Atlantidamin.flatten()
})

# Cargar el archivo GeoJSON
geojson_pathAPoligono = 'BrianGallardo/WRF-HN-ClimaScope/15-11-2023 12_00/GeoJson/Departamentos/Atlantida.geojson'
APoligono_gdf = gpd.read_file(geojson_pathAPoligono) #RAPavg_ (Raster Atlantida Precipitacio Promedio)

# Extraer las coordenadas del contorno
# Extraer las coordenadas para el contorno
x_geojsonPmin, y_geojsonPmin = [], []
for geom in APoligono_gdf.geometry:
    if geom.geom_type == 'Polygon':
        x, y = geom.exterior.coords.xy
        x_geojsonPmin.extend(x)
        y_geojsonPmin.extend(y)
    elif geom.geom_type == 'MultiPolygon':
        for part in geom.geoms:  # Iterar sobre cada polígono en el MultiPolygon
            x, y = part.exterior.coords.xy
            x_geojsonPmin.extend(x)
            y_geojsonPmin.extend(y)

# Crear un gráfico con los ejes georreferenciados
fig_map_Atlantidamin = px.imshow(
    dfAtlanPPminRaster.pivot(index='y', columns='x', values='value'),
    color_continuous_scale='Blues',
    labels={'color': 'Precipitación (mm)'},
    origin='lower'  # Esto asegura que el gráfico se alinee con las coordenadas geográficas
)

# Añadir la traza del contorno
fig_map_Atlantidamin.add_trace(go.Scatter(x=x_geojsonPmin, y=y_geojsonPmin, mode='lines', name='Atlantida',
                         line=dict(color='black', width=1)))

fig_map_Atlantidamin.update_layout(
    title_text='Precipitación Mínima en Atlántida del Pronóstico WRF',
    title_x=0.5,
    title_font=dict(size=14),
    height=500,
    width=1320,
    margin=dict(t=40, b=0, l=0, r=0),
    coloraxis_colorbar=dict(
        title='PP (mm)',
        titleside='right',
        thicknessmode='pixels', thickness=15,
        lenmode='pixels', len=300,
        yanchor='middle', y=0.5,
        titlefont=dict(size=12)
    ),
    xaxis_showgrid=False,  # Oculta la cuadrícula del eje X
    yaxis_showgrid=False   # Oculta la cuadrícula del eje Y
)

#############################################################################
raster_file_Atlantidamax = 'BrianGallardo/WRF-HN-ClimaScope/15-11-2023 12_00/Rasters/Departamentos/Atlántida/max_dia5.tif'
with rasterio.open(raster_file_Atlantidamax) as src:
    raster_array_Atlantidamax = src.read(1)
    # Obtener transformación de Rasterio para convertir coordenadas de píxeles a geográficas
    transform = src.transform

    # Calcular las coordenadas geográficas para cada píxel
    height, width = raster_array_Atlantidamax.shape
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')

# Crear un DataFrame para almacenar las coordenadas y los valores
dfAtlanPPmaxRaster = pd.DataFrame({
    'x': np.array(xs).flatten(),
    'y': np.array(ys).flatten(),
    'value': raster_array_Atlantidamax.flatten()
})

# Cargar el archivo GeoJSON
geojson_pathAPoligono = 'BrianGallardo/WRF-HN-ClimaScope/15-11-2023 12_00/GeoJson/Departamentos/Atlantida.geojson'
APoligono_gdf = gpd.read_file(geojson_pathAPoligono) #RAPavg_ (Raster Atlantida Precipitacio Promedio)

# Extraer las coordenadas del contorno
# Extraer las coordenadas para el contorno
x_geojsonPmax, y_geojsonPmax = [], []
for geom in APoligono_gdf.geometry:
    if geom.geom_type == 'Polygon':
        x, y = geom.exterior.coords.xy
        x_geojsonPmax.extend(x)
        y_geojsonPmax.extend(y)
    elif geom.geom_type == 'MultiPolygon':
        for part in geom.geoms:  # Iterar sobre cada polígono en el MultiPolygon
            x, y = part.exterior.coords.xy
            x_geojsonPmax.extend(x)
            y_geojsonPmax.extend(y)

# Crear un gráfico con los ejes georreferenciados
fig_map_Atlantidamax = px.imshow(
    dfAtlanPPmaxRaster.pivot(index='y', columns='x', values='value'),
    color_continuous_scale='Blues',
    labels={'color': 'Precipitación (mm)'},
    origin='lower'  # Esto asegura que el gráfico se alinee con las coordenadas geográficas
)

# Añadir la traza del contorno
fig_map_Atlantidamax.add_trace(go.Scatter(x=x_geojsonPmax, y=y_geojsonPmax, mode='lines', name='Atlantida',
                         line=dict(color='black', width=1)))


fig_map_Atlantidamax.update_layout(
    title_text='Precipitación Máxima en Atlántida del Pronóstico WRF',
    title_x=0.5,
    title_font=dict(size=14),
    height=500,
    width=1320,
    margin=dict(t=40, b=0, l=0, r=0),
    coloraxis_colorbar=dict(
        title='PP (mm)',
        titleside='right',
        thicknessmode='pixels', thickness=15,
        lenmode='pixels', len=300,
        yanchor='middle', y=0.5,
        titlefont=dict(size=12)
    ),
    xaxis_showgrid=False,  # Oculta la cuadrícula del eje X
    yaxis_showgrid=False   # Oculta la cuadrícula del eje Y
)
############################################# MAPAS 2D

#................Mapa Precipitacion Promedio 3D WRF............
# Carga el archivo raster para cada Departamento
raster_file_Atlantidaavg = 'BrianGallardo/WRF-HN-ClimaScope/15-11-2023 12_00/Rasters/Departamentos/Atlántida/avg_dia5.tif'

with rasterio.open(raster_file_Atlantidaavg) as src:
    raster_array_Atlantidaavg = src.read(1)
    transform = src.transform

    # Obtener las dimensiones del raster
    height, width = raster_array_Atlantidaavg.shape

    # Generar una cuadrícula de coordenadas basada en las dimensiones del raster
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    x, y = np.meshgrid(x, y)

# Generar coordenadas geográficas para cada píxel
    height, width = raster_array_Atlantidaavg.shape
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    x_geo, y_geo = rasterio.transform.xy(transform, rows.flatten(), cols.flatten())
    x_geo = np.array(x_geo).reshape((height, width))
    y_geo = np.array(y_geo).reshape((height, width))

def transform_geojson_to_3davg(Atlantidaavg3d_gdf):
    # Extraer las coordenadas del contorno exterior de las geometrías
    x_geojsonPavg3D, y_geojsonPavg3D = Atlantidaavg3d_gdf.geometry.exterior.coords.xy
    z_geojsonPavg3D = [0] * len(x_geojsonPavg3D)  # Altura constante para el contorno
    return x_geojsonPavg3D, y_geojsonPavg3D, z_geojsonPavg3D

# Suponiendo que tienes un archivo GeoJSON para la región
geojson_pathPavg3D = 'BrianGallardo/WRF-HN-ClimaScope/15-11-2023 12_00/GeoJson/Departamentos/Atlantida.geojson'
Atlantidaavg3d_gdf = gpd.read_file(geojson_pathPavg3D)
# Transformar las coordenadas GeoJSON para el gráfico 3D
# Proyectar a un CRS adecuado (por ejemplo, UTM)
Atlantidaavg3d_gdf = Atlantidaavg3d_gdf.to_crs(epsg=4276) # Cambia el EPSG según tu zona (para WRF es 4276)
# Ahora puedes calcular los centroides
# Extraer las coordenadas para el contorno
x_geojsonPavg3D, y_geojsonPavg3D = [], []
for geom in Atlantidaavg3d_gdf.geometry:
    if geom.geom_type == 'Polygon':
        x, y = geom.exterior.coords.xy
        x_geojsonPavg3D.extend(x)
        y_geojsonPavg3D.extend(y)
    elif geom.geom_type == 'MultiPolygon':
        for part in geom.geoms:  # Usa 'geoms' para iterar sobre los polígonos en un MultiPolygon
            x, y = part.exterior.coords.xy
            x_geojsonPavg3D.extend(x)
            y_geojsonPavg3D.extend(y)

z_geojsonPavg3D = [27] * len(x_geojsonPavg3D)  # Altura constante para el contorno

# Crear un gráfico de Precipitacion 3D
fig_map_Atlantidaavg3d = go.Figure(data=[go.Surface(
    z=raster_array_Atlantidaavg, x=x_geo, y=y_geo, colorscale='Blues',
    cmin=np.nanmin(raster_array_Atlantidaavg), cmax=np.nanmax(raster_array_Atlantidaavg),
    #opacity=0.9,
    name='z:PP(mm)'
    )])

# Añadir el gráfico raster
# Añadir trazas de puntos o líneas 3D, sin la propiedad 'contours'
fig_map_Atlantidaavg3d.add_trace(go.Scatter3d(
    x=x_geojsonPavg3D,
    y=y_geojsonPavg3D,
    z=z_geojsonPavg3D,
    mode='lines',  # o 'markers'
    line=dict(color='cyan', width=2)  # Estilos para la línea del contorno
    # Otras propiedades como color, tamaño, etc.
))

#Precipitacion 3D
# Aplicar contornos solo a la traza de superficie
fig_map_Atlantidaavg3d.update_traces(
    selector=dict(name='z:PP(mm)'),  # Seleccionar la traza por su nombre
    contours_z=dict(
        show=True, usecolormap=True, highlightcolor="limegreen", project_z=True
    )
)

# Actualizar el diseño del gráfico
fig_map_Atlantidaavg3d.update_layout(
    title='Precipitación Promedio en Atlántida del Pronóstico WRF',
    template='plotly_dark',
    autosize=False,
    width=1320,
    height=600,
    margin=dict(t=50, l=20, r=50, b=30),
scene=dict(
    xaxis=dict(title='Longitud'),
    yaxis=dict(title='Latitud'),
    zaxis=dict(title='Precipitación (mm)', nticks=4),
    aspectratio=dict(x=6, y=1.5, z=0.2),
camera=dict(
    eye=dict(x=6, y=-8, z=0.2)),
    xaxis_showgrid=True, # Ocultar la cuadrícula para una vista más limpia
    yaxis_showgrid=True,
    zaxis_showgrid=False)) # Puedes ajustar estos valores para cambiar el aspecto de la visualización 3D

coloraxis_colorbar=dict(
    title='PP (mm)',
    titleside='right',
    thicknessmode='pixels', thickness=15,
    lenmode='pixels', len=200,
    yanchor='middle', y=0.5,
    titlefont=dict(size=12),
    xaxis=dict(title='Longitud'),
    yaxis=dict(title='Latitud'),
    zaxis=dict(title='Precipitación (mm)'))

fig_map_Atlantidaavg3d.update_xaxes(
tickangle=20,
title_standoff=10
)

fig_map_Atlantidaavg3d.update_yaxes(
tickangle=120,
title_standoff=25
)
#############################################
#...........Mapa Precipitacion Acumulada 3D WRF...........
# Carga el archivo raster para cada Departamento
raster_file_Atlantidaac = 'BrianGallardo/WRF-HN-ClimaScope/15-11-2023 12_00/Rasters/Departamentos/Atlántida/ppAcum5dias.tif'

with rasterio.open(raster_file_Atlantidaac) as src:
    raster_array_Atlantidaac = src.read(1)
    transform = src.transform

    # Obtener las dimensiones del raster
    height, width = raster_array_Atlantidaac.shape

    # Generar una cuadrícula de coordenadas basada en las dimensiones del raster
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    x, y = np.meshgrid(x, y)

# Generar coordenadas geográficas para cada píxel
    height, width = raster_array_Atlantidaac.shape
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    x_geo, y_geo = rasterio.transform.xy(transform, rows.flatten(), cols.flatten())
    x_geo = np.array(x_geo).reshape((height, width))
    y_geo = np.array(y_geo).reshape((height, width))

def transform_geojson_to_3dac(Atlantidaac3d_gdf):
    # Extraer las coordenadas del contorno exterior de las geometrías
    x_geojsonac, y_geojsonac = Atlantidaac3d_gdf.geometry.exterior.coords.xy
    z_geojsonac = [0] * len(x_geojsonac)  # Altura constante para el contorno
    return x_geojsonac, y_geojsonac, z_geojsonac

# Suponiendo que tienes un archivo GeoJSON para la región
geojson_pathac = 'BrianGallardo/WRF-HN-ClimaScope/15-11-2023 12_00/GeoJson/Departamentos/Atlantida.geojson'
Atlantidaac3d_gdf = gpd.read_file(geojson_pathac)
# Transformar las coordenadas GeoJSON para el gráfico 3D
# Proyectar a un CRS adecuado (por ejemplo, UTM)
Atlantidaac3d_gdf = Atlantidaac3d_gdf.to_crs(epsg=4276) # Cambia el EPSG según tu zona (para WRF es 4276)
# Ahora puedes calcular los centroides
# Extraer las coordenadas para el contorno
x_geojsonac, y_geojsonac = [], []
for geom in Atlantidaac3d_gdf.geometry:
    if geom.geom_type == 'Polygon':
        x, y = geom.exterior.coords.xy
        x_geojsonac.extend(x)
        y_geojsonac.extend(y)
    elif geom.geom_type == 'MultiPolygon':
        for part in geom.geoms:  # Usa 'geoms' para iterar sobre los polígonos en un MultiPolygon
            x, y = part.exterior.coords.xy
            x_geojsonac.extend(x)
            y_geojsonac.extend(y)

z_geojsonac = [60] * len(x_geojsonac)  # Altura constante para el contorno

# Crear un gráfico de Precipitacion 3D
fig_map_Atlantidaac3d = go.Figure(data=[go.Surface(
    z=raster_array_Atlantidaac, x=x_geo, y=y_geo, colorscale='Blues',
    cmin=np.nanmin(raster_array_Atlantidaac), cmax=np.nanmax(raster_array_Atlantidaac),
    #opacity=0.9,
    name='z:PP(mm)'
    )])

# Añadir el gráfico raster
# Añadir trazas de puntos o líneas 3D, sin la propiedad 'contours'
fig_map_Atlantidaac3d.add_trace(go.Scatter3d(
    x=x_geojsonac,
    y=y_geojsonac,
    z=z_geojsonac,
    mode='lines',  # o 'markers'
    line=dict(color='cyan', width=2)  # Estilos para la línea del contorno
    # Otras propiedades como color, tamaño, etc.
))

#Precipitacion 3D
# Aplicar contornos solo a la traza de superficie
fig_map_Atlantidaac3d.update_traces(
    selector=dict(name='z:PP(mm)'),  # Seleccionar la traza por su nombre
    contours_z=dict(
        show=True, usecolormap=True, highlightcolor="limegreen", project_z=True
    )
)


# Actualizar el diseño del gráfico
fig_map_Atlantidaac3d.update_layout(
    title='Precipitación Acumulada en Atlántida del Pronóstico WRF',
    template='plotly_dark',
    autosize=False,
    width=1320,
    height=600,
    margin=dict(t=50, l=20, r=50, b=30),
scene=dict(
    xaxis=dict(title='Longitud'),
    yaxis=dict(title='Latitud'),
    zaxis=dict(title='Precipitación (mm)', nticks=4),
    aspectratio=dict(x=6, y=1.5, z=0.2),
camera=dict(
    eye=dict(x=6, y=-8, z=0.2)),
    xaxis_showgrid=True, # Ocultar la cuadrícula para una vista más limpia
    yaxis_showgrid=True,
    zaxis_showgrid=False)) # Puedes ajustar estos valores para cambiar el aspecto de la visualización 3D

coloraxis_colorbar=dict(
    title='PP (mm)',
    titleside='right',
    thicknessmode='pixels', thickness=15,
    lenmode='pixels', len=200,
    yanchor='middle', y=0.5,
    titlefont=dict(size=12),
    xaxis=dict(title='Longitud'),
    yaxis=dict(title='Latitud'),
    zaxis=dict(title='Precipitación (mm)'))

fig_map_Atlantidaac3d.update_xaxes(
tickangle=20,
title_standoff=10
)

fig_map_Atlantidaac3d.update_yaxes(
tickangle=120,
title_standoff=25
)

#############################################
#...........Mapa Precipitacion Minima 3D WRF...........
# Carga el archivo raster para cada Departamento
raster_file_Atlantidamin = 'BrianGallardo/WRF-HN-ClimaScope/15-11-2023 12_00/Rasters/Departamentos/Atlántida/min_dia5.tif'

with rasterio.open(raster_file_Atlantidamin) as src:
    raster_array_Atlantidamin = src.read(1)
    transform = src.transform

    # Obtener las dimensiones del raster
    height, width = raster_array_Atlantidamin.shape

    # Generar una cuadrícula de coordenadas basada en las dimensiones del raster
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    x, y = np.meshgrid(x, y)

# Generar coordenadas geográficas para cada píxel
    height, width = raster_array_Atlantidamin.shape
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    x_geo, y_geo = rasterio.transform.xy(transform, rows.flatten(), cols.flatten())
    x_geo = np.array(x_geo).reshape((height, width))
    y_geo = np.array(y_geo).reshape((height, width))

def transform_geojson_to_3dmin(Atlantidamin3d_gdf):
    # Extraer las coordenadas del contorno exterior de las geometrías
    x_geojsonmin, y_geojsonmin = Atlantidamin3d_gdf.geometry.exterior.coords.xy
    z_geojsonmin = [0] * len(x_geojsonmin)  # Altura constante para el contorno
    return x_geojsonmin, y_geojsonmin, z_geojsonmin

# Suponiendo que tienes un archivo GeoJSON para la región
geojson_pathmin = 'BrianGallardo/WRF-HN-ClimaScope/15-11-2023 12_00/GeoJson/Departamentos/Atlantida.geojson'
Atlantidamin3d_gdf = gpd.read_file(geojson_pathmin)
# Transformar las coordenadas GeoJSON para el gráfico 3D
# Proyectar a un CRS adecuado (por ejemplo, UTM)
Atlantidamin3d_gdf = Atlantidamin3d_gdf.to_crs(epsg=4276) # Cambia el EPSG según tu zona (para WRF es 4276)
# Ahora puedes calcular los centroides
# Extraer las coordenadas para el contorno
x_geojsonmin, y_geojsonmin = [], []
for geom in Atlantidamin3d_gdf.geometry:
    if geom.geom_type == 'Polygon':
        x, y = geom.exterior.coords.xy
        x_geojsonmin.extend(x)
        y_geojsonmin.extend(y)
    elif geom.geom_type == 'MultiPolygon':
        for part in geom.geoms:  # Usa 'geoms' para iterar sobre los polígonos en un MultiPolygon
            x, y = part.exterior.coords.xy
            x_geojsonmin.extend(x)
            y_geojsonmin.extend(y)

z_geojsonmin = [27] * len(x_geojsonmin)  # Altura constante para el contorno

# Crear un gráfico de Precipitacion 3D
fig_map_Atlantidamin3d = go.Figure(data=[go.Surface(
    z=raster_array_Atlantidamin, x=x_geo, y=y_geo, colorscale='Blues',
    cmin=np.nanmin(raster_array_Atlantidamin), cmax=np.nanmax(raster_array_Atlantidamin),
    #opacity=0.9,
    name='z:PP(mm)'
    )])

# Añadir el gráfico raster
# Añadir trazas de puntos o líneas 3D, sin la propiedad 'contours'
fig_map_Atlantidamin3d.add_trace(go.Scatter3d(
    x=x_geojsonmin,
    y=y_geojsonmin,
    z=z_geojsonmin,
    mode='lines',  # o 'markers'
    line=dict(color='cyan', width=2)  # Estilos para la línea del contorno
    # Otras propiedades como color, tamaño, etc.
))

#Precipitacion 3D
# Aplicar contornos solo a la traza de superficie
fig_map_Atlantidamin3d.update_traces(
    selector=dict(name='z:PP(mm)'),  # Seleccionar la traza por su nombre
    contours_z=dict(
        show=True, usecolormap=True, highlightcolor="limegreen", project_z=True
    )
)


# Actualizar el diseño del gráfico
fig_map_Atlantidamin3d.update_layout(
    title='Precipitación Mínima en Atlántida del Pronóstico WRF',
    template='plotly_dark',
    autosize=False,
    width=1320,
    height=600,
    margin=dict(t=50, l=20, r=50, b=30),
scene=dict(
    xaxis=dict(title='Longitud'),
    yaxis=dict(title='Latitud'),
    zaxis=dict(title='Precipitación (mm)', nticks=4),
    aspectratio=dict(x=6, y=1.5, z=0.2),
camera=dict(
    eye=dict(x=6, y=-8, z=0.2)),
    xaxis_showgrid=True, # Ocultar la cuadrícula para una vista más limpia
    yaxis_showgrid=True,
    zaxis_showgrid=False)) # Puedes ajustar estos valores para cambiar el aspecto de la visualización 3D

coloraxis_colorbar=dict(
    title='PP (mm)',
    titleside='right',
    thicknessmode='pixels', thickness=15,
    lenmode='pixels', len=200,
    yanchor='middle', y=0.5,
    titlefont=dict(size=12),
    xaxis=dict(title='Longitud'),
    yaxis=dict(title='Latitud'),
    zaxis=dict(title='Precipitación (mm)'))

fig_map_Atlantidamin3d.update_xaxes(
tickangle=20,
title_standoff=10
)

fig_map_Atlantidamin3d.update_yaxes(
tickangle=120,
title_standoff=25
)
#############################################
#...........Mapa Precipitacion Maxima 3D WRF...........
# Carga el archivo raster para cada Departamento
raster_file_Atlantidamax = 'BrianGallardo/WRF-HN-ClimaScope/15-11-2023 12_00/Rasters/Departamentos/Atlántida/max_dia5.tif'

with rasterio.open(raster_file_Atlantidamax) as src:
    raster_array_Atlantidamax = src.read(1)
    transform = src.transform

    # Obtener las dimensiones del raster
    height, width = raster_array_Atlantidamax.shape

    # Generar una cuadrícula de coordenadas basada en las dimensiones del raster
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    x, y = np.meshgrid(x, y)

# Generar coordenadas geográficas para cada píxel
    height, width = raster_array_Atlantidamax.shape
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    x_geo, y_geo = rasterio.transform.xy(transform, rows.flatten(), cols.flatten())
    x_geo = np.array(x_geo).reshape((height, width))
    y_geo = np.array(y_geo).reshape((height, width))

def transform_geojson_to_3dmax(Atlantidamax3d_gdf):
    # Extraer las coordenadas del contorno exterior de las geometrías
    x_geojsonmax, y_geojsonmax = Atlantidamax3d_gdf.geometry.exterior.coords.xy
    z_geojsonmax = [0] * len(x_geojsonmax)  # Altura constante para el contorno
    return x_geojsonmax, y_geojsonmax, z_geojsonmax

# Suponiendo que tienes un archivo GeoJSON para la región
geojson_pathmax = 'BrianGallardo/WRF-HN-ClimaScope/15-11-2023 12_00/GeoJson/Departamentos/Atlantida.geojson'
Atlantidamax3d_gdf = gpd.read_file(geojson_pathmax)
# Transformar las coordenadas GeoJSON para el gráfico 3D
# Proyectar a un CRS adecuado (por ejemplo, UTM)
Atlantidamax3d_gdf = Atlantidamax3d_gdf.to_crs(epsg=4276) # Cambia el EPSG según tu zona (para WRF es 4276)
# Ahora puedes calcular los centroides
# Extraer las coordenadas para el contorno
x_geojsonmax, y_geojsonmax = [], []
for geom in Atlantidamax3d_gdf.geometry:
    if geom.geom_type == 'Polygon':
        x, y = geom.exterior.coords.xy
        x_geojsonmax.extend(x)
        y_geojsonmax.extend(y)
    elif geom.geom_type == 'MultiPolygon':
        for part in geom.geoms:  # Usa 'geoms' para iterar sobre los polígonos en un MultiPolygon
            x, y = part.exterior.coords.xy
            x_geojsonmax.extend(x)
            y_geojsonmax.extend(y)

z_geojsonmax = [32] * len(x_geojsonmax)  # Altura constante para el contorno

# Crear un gráfico de Precipitacion 3D
fig_map_Atlantidamax3d = go.Figure(data=[go.Surface(
    z=raster_array_Atlantidamax, x=x_geo, y=y_geo, colorscale='Blues',
    cmin=np.nanmin(raster_array_Atlantidamax), cmax=np.nanmax(raster_array_Atlantidamax),
    #opacity=0.9,
    name='z:PP(mm)'
    )])

# Añadir el gráfico raster
# Añadir trazas de puntos o líneas 3D, sin la propiedad 'contours'
fig_map_Atlantidamax3d.add_trace(go.Scatter3d(
    x=x_geojsonmax,
    y=y_geojsonmax,
    z=z_geojsonmax,
    mode='lines',  # o 'markers'
    line=dict(color='cyan', width=2)  # Estilos para la línea del contorno
    # Otras propiedades como color, tamaño, etc.
))

#Precipitacion 3D
# Aplicar contornos solo a la traza de superficie
fig_map_Atlantidamax3d.update_traces(
    selector=dict(name='z:PP(mm)'),  # Seleccionar la traza por su nombre
    contours_z=dict(
        show=True, usecolormap=True, highlightcolor="limegreen", project_z=True
    )
)


# Actualizar el diseño del gráfico
fig_map_Atlantidamax3d.update_layout(
    title='Precipitación Máxima en Atlántida del Pronóstico WRF',
    template='plotly_dark',
    autosize=False,
    width=1320,
    height=600,
    margin=dict(t=50, l=20, r=50, b=30),
scene=dict(
    xaxis=dict(title='Longitud'),
    yaxis=dict(title='Latitud'),
    zaxis=dict(title='Precipitación (mm)', nticks=4),
    aspectratio=dict(x=6, y=1.5, z=0.2),
camera=dict(
    eye=dict(x=6, y=-8, z=0.2)),
    xaxis_showgrid=True, # Ocultar la cuadrícula para una vista más limpia
    yaxis_showgrid=True,
    zaxis_showgrid=False)) # Puedes ajustar estos valores para cambiar el aspecto de la visualización 3D

coloraxis_colorbar=dict(
    title='PP (mm)',
    titleside='right',
    thicknessmode='pixels', thickness=15,
    lenmode='pixels', len=200,
    yanchor='middle', y=0.5,
    titlefont=dict(size=12),
    xaxis=dict(title='Longitud'),
    yaxis=dict(title='Latitud'),
    zaxis=dict(title='Precipitación (mm)'))

fig_map_Atlantidamax3d.update_xaxes(
tickangle=20,
title_standoff=10
)

fig_map_Atlantidamax3d.update_yaxes(
tickangle=120,
title_standoff=25
)


#..........Termina La Seccion Departamentos................
dropdown_menu_style = {
    'maxHeight': '300px',  # Limita la altura máxima del menú desplegable
    'overflowY': 'auto'    # Aplica desplazamiento vertical si es necesario
}

# Establece la app Dash
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.CERULEAN],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1'}])
server = app.server
app.title = "WRF-HN ClimaScope"
app.index_string = open('BrianGallardo/WRF-HN-ClimaScope/WRF-HN ClimaScope/template/template.html', 'r').read()
spinner_wrapper = html.Div(
    dbc.Spinner(type="grow", color="dark", children=[html.Div(id="spinner-output")]),
    id="spinner-wrapper"
)
footer = html.Footer(
    html.P(
        ["© ", html.Span(id="current-year"), " Visualizar datos meteorológicos y geográficos pronosticados por el modelo WRF (Weather Research and Forecasting). Todos los derechos reservados."],
        className='footer',  # Aplica la clase 'footer'
        style={'textAlign': 'center'}
    ),
    style={'backgroundColor': '#00771C', 'color': 'white', 'padding': '10px',}
)
# Define la interfaz de usuario
app.layout = html.Div([
    html.H1(["", dbc.Badge("WRF-HN ClimaScope", className="my-custom-class")], style={'textAlign': 'center', 'height': 'auto', 'width': 'auto', 'backgroundColor': '#00771C'}),
    # Barra de navegacion
    html.Nav(
        children=[
            html.Img(src='/assets/LOGO.png', style={'height': '100px', 'width': 'auto'}),  # Ajusta el tamaño como necesites
    html.A(
        className='navbar-brand',
        href='WRF_HN_ClimaScope',
        id='link-ClimaScope',
        style={
        'marginLeft': '1px',
        'height': '55px',
        'width': '140px',
        'backgroundColor': '#269523',
        'borderRadius': '15px',  # Ajusta este valor para controlar el redondeo
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'center',
        'paddingLeft': '10px',  # Aumenta el espacio a la izquierda
        'paddingRight': '10px', # Aumenta el espacio a la derecha
        'textDecoration': 'none',  # Opcional, para eliminar el subrayado del enlace
        'color': 'white'  # Opcional, para cambiar el color del texto
    },
    children='ClimaScope'
    ),
            html.Div(
                className='collapse navbar-collapse',
                id='navbarNav',
            ),
            dcc.Location(id='url', refresh=False, ),
            dbc.DropdownMenu(
                className='btn-outline', style={'marginLeft': '40px'},
            children=[
                html.Div([
                dbc.DropdownMenuItem("País", header=True),
                dbc.DropdownMenuItem(divider=True),
                dbc.DropdownMenuItem("Honduras", href="/WRF_pais_Honduras", id='link-Pais', n_clicks=0),
                # Añadir más departamentos aquí...
                ], style=dropdown_menu_style)  # Aplicando el estilo aquí
            ],
            nav=True,
            in_navbar=True,
            label="País",
            toggle_style={
                "textTransform": "uppercase",
                "borderRadius": "25px", # Hace los bordes más redondeados para una forma más ovalada
                "background": "rgba(38, 149, 35, 1)",  # 0.1-1 Aumenta la transparencia
                "padding": "10px 20px",  # Ajusta el relleno para que sea más amplio y ovalado
                'display': 'inline',
                'textDecoration': 'none',  # Opcional, para eliminar el subrayado del enlace
                'color': 'white'  # Opcional, para cambiar el color del texto
                },
            toggleClassName="fst-sans border border-dark",
        ),
            dbc.DropdownMenu(
                className='btn-outline', style={'marginLeft': '5px'},
            children=[
                html.Div([
                dbc.DropdownMenuItem("Departamentos de Honduras", header=True),
                dbc.DropdownMenuItem(divider=True),
                dbc.DropdownMenuItem("Atlántida", href="/departamento_WRF_atlantida", id='link-Departamentos', n_clicks=0),
                dbc.DropdownMenuItem("Colón", href="#departamento2"),
                dbc.DropdownMenuItem("Comayagua", href="#departamento2"),
                dbc.DropdownMenuItem("Copán", href="#departamento2"),
                dbc.DropdownMenuItem("Cortés", href="#departamento2"),
                dbc.DropdownMenuItem("Choluteca", href="#departamento2"),
                dbc.DropdownMenuItem("El Paraíso", href="#departamento2"),
                dbc.DropdownMenuItem("Francisco Morazán", href="#departamento2"),
                dbc.DropdownMenuItem("Gracias a Dios", href="#departamento2"),
                dbc.DropdownMenuItem("Intibucá", href="#departamento2"),
                dbc.DropdownMenuItem("Islas de la Bahía", href="#departamento2"),
                dbc.DropdownMenuItem("La Paz", href="#departamento2"),
                dbc.DropdownMenuItem("Lempira", href="#departamento2"),
                dbc.DropdownMenuItem("Ocotepeque", href="#departamento2"),
                dbc.DropdownMenuItem("Olancho", href="#departamento2"),
                dbc.DropdownMenuItem("Santa Bárbara", href="#departamento2"),
                dbc.DropdownMenuItem("Valle", href="#departamento2"),
                dbc.DropdownMenuItem("Yoro", href="#departamento2"),
                # Añadir más departamentos aquí...
                ], style=dropdown_menu_style)  # Aplicando el estilo aquí
            ],
            nav=True,
            in_navbar=True,
            label="Departamentos",
            toggle_style={
                "textTransform": "uppercase",
                "borderRadius": "25px", # Hace los bordes más redondeados para una forma más ovalada
                "background": "rgba(38, 149, 35, 1)",  # Aumenta la transparencia
                "padding": "10px 20px",  # Ajusta el relleno para que sea más amplio y ovalado
                'display': 'inline',
                'textDecoration': 'none',  # Opcional, para eliminar el subrayado del enlace
                'color': 'white'  # Opcional, para cambiar el color del texto
                },
            toggleClassName="fst-sans border border-dark",
        ),
            dbc.DropdownMenu(
                className='btn-outline', style={'marginLeft': '5px'},
            children=[
                html.Div([
                dbc.DropdownMenuItem("Municipios del Departamen de Atlántida", header=True),
                dbc.DropdownMenuItem("Atlántida", href="#atlantida", id='link-Municipios'),
                dbc.DropdownMenuItem("Municipio 2", href="#Atlantida_Municipio_WRF_"),
                dbc.DropdownMenuItem("Municipio 3", href="#Atlantida_Municipio_WRF_"),
                dbc.DropdownMenuItem("Municipio 4", href="#Atlantida_Municipio_WRF_"),
                dbc.DropdownMenuItem("Municipio 5", href="#Atlantida_Municipio_WRF_"),
                dbc.DropdownMenuItem(divider=True),
                dbc.DropdownMenuItem("Municipios del Departamen de Cortes", header=True),
                # Añadir más departamentos aquí...
                dbc.DropdownMenuItem("Municipio 298", href="#Atlantida_Municipio_WRF_"),
                dbc.DropdownMenuItem("Departamento 18", href="#departamento18"),
                ], style=dropdown_menu_style)  # Aplicando el estilo aquí
            ],
            nav=True,
            in_navbar=True,
            label="Municipios",
            toggle_style={
                "textTransform": "uppercase",
                "borderRadius": "25px", # Hace los bordes más redondeados para una forma más ovalada
                "background": "rgba(38, 149, 35, 1)",  # Aumenta la transparencia
                "padding": "10px 20px",  # Ajusta el relleno para que sea más amplio y ovalado
                'display': 'inline',
                'textDecoration': 'none',  # Opcional, para eliminar el subrayado del enlace
                'color': 'white'  # Opcional, para cambiar el color del texto
                },
            toggleClassName="fst-sans border border-dark",
        ),
            dbc.DropdownMenu(
                className='btn-outline', style={'marginLeft': '5px'},
            children=[
                html.Div([
                dbc.DropdownMenuItem("Cuencas Hidrográficas de Honduras", header=True),
                dbc.DropdownMenuItem("Cuenca Aguan", href="#Aguan", id='link-Cuencas'),
                dbc.DropdownMenuItem("Cuenca 2", href="#Cuencas_WRF_"),
                dbc.DropdownMenuItem("Cuenca 3", href="#Cuencas_WRF_"),
                dbc.DropdownMenuItem("Cuenca 4", href="#Cuencas_WRF_"),
                dbc.DropdownMenuItem("Cuenca 5", href="#Cuencas_WRF_"),
                # Añadir más departamentos aquí...
                dbc.DropdownMenuItem("Cuenca 25", href="#Cuencas_WRF_"),
                ], style=dropdown_menu_style)  # Aplicando el estilo aquí
            ],
            nav=True,
            in_navbar=True,
            label="Cuencas Hidrográficas",
            toggle_style={
                "textTransform": "uppercase",
                "borderRadius": "25px", # Hace los bordes más redondeados para una forma más ovalada
                "background": "rgba(38, 149, 35, 1)",  # Aumenta la transparencia
                "padding": "10px 20px",  # Ajusta el relleno para que sea más amplio y ovalado
                'display': 'inline',
                'textDecoration': 'none',  # Opcional, para eliminar el subrayado del enlace
                'color': 'white'  # Opcional, para cambiar el color del texto
                },
            toggleClassName="fst-sans border border-dark",
        ),
            dbc.DropdownMenu(
                className='btn-outline', style={'marginLeft': '5px'},
            children=[
                html.Div([
                dbc.DropdownMenuItem("Subcuencas Hidrográficas de la Cuenca Aguan", header=True),
                dbc.DropdownMenuItem("Aguan Alto", href="#Aguan_Alto", id='link-Subcuencas'),
                dbc.DropdownMenuItem("Subcuenca 2", href="#Subcuencas_WRF_"),
                dbc.DropdownMenuItem("Subcuenca 3", href="#Subcuencas_WRF_"),
                dbc.DropdownMenuItem("Subcuenca 4", href="#Subcuencas_WRF_"),
                dbc.DropdownMenuItem("Subcuenca 5", href="#Subcuencas_WRF_"),
                dbc.DropdownMenuItem(divider=True),
                dbc.DropdownMenuItem("Subcuencas de la Cuenca Choluteca", header=True),
                # Añadir más departamentos aquí...
                dbc.DropdownMenuItem("Subcuenca 133", href="#Subcuencas_WRF_"),
                ], style=dropdown_menu_style)  # Aplicando el estilo aquí
            ],
            nav=True,
            in_navbar=True,
            label="Subcuencas Hidrográficas",
            toggle_style={
                "textTransform": "uppercase",
                "borderRadius": "25px", # Hace los bordes más redondeados para una forma más ovalada
                "background": "rgba(38, 149, 35, 1)",  # Aumenta la transparencia
                "padding": "10px 20px",  # Ajusta el relleno para que sea más amplio y ovalado
                'display': 'inline',
                'textDecoration': 'none',  # Opcional, para eliminar el subrayado del enlace
                'color': 'white'  # Opcional, para cambiar el color del texto
                },
            toggleClassName="fst-sans border border-dark",
        ),
            # Tooltips
        #dbc.Tooltip("Información sobre País", target="tooltip-pais"),
        #dbc.Tooltip("Información sobre Departamentos", target="tooltip-departamentos"),
# ... [otros tooltips para otros elementos] ...
            dbc.Container()
            #dbc.Row(
            #   [
            #      dbc.Col(dbc.Input(id="search-input", type="search", placeholder="Buscar")),
            #     dbc.Col(
                #        dbc.Button("Buscar", id="search-button", color="success", className="ms-2", n_clicks=0),
                #        width="auto",
                #    ),
                #],
                #className="g-0 ms-auto flex-nowrap mt-2 mt-md-0",
                #align="center",
            #),
        ],
        className='navbar navbar-expand-lg ml-auto navbar-dark', # Clases de Bootstrap
        style={'backgroundColor': '#00771C'},  # Estilo en línea para el color de fondo
    ),
    # Este Div mostrará los resultados de la búsqueda
    # Sección Atlantida
    html.Br(),
    html.Div(id="page-content"),  # Este Div contendrá el contenido de la página seleccionada
    html.Br(),
    spinner_wrapper,  # spinner
    html.Br(),
    html.Div(id='main-content'),
    html.Br(),
    # Pie de página
    footer
])


# Devolución de llamada para sincronizar la nueva gráfica con las filas seleccionadas en la tabla Atlantida
# Callback optimizado
# Callback para manejar los clics en los enlaces de navegación
@app.callback(
    Output('page-content', 'children'),
    [Input('link-Pais', 'n_clicks'),
    Input('link-Departamentos', 'n_clicks'),
    Input('link-Municipios', 'n_clicks'),
    Input('link-Cuencas', 'n_clicks'),
    Input('link-Subcuencas', 'n_clicks')],  # Asumiendo que 'navbarNav' es el componente que captura la navegación
    prevent_initial_call = True
# Esto debería ser el estado actual de la navegación, tal vez necesites cambiarlo por algo más específico

)
def navigate_page(*args):
    ctx = dash.callback_context
    print(ctx.triggered)  # Agrega esta línea para depurar
    if not ctx.triggered:
        # Si no se ha clickeado ningún enlace, se puede devolver el contenido predeterminado
        return html.Div("Selecciona una sección de la barra de navegación.", style={'font-weight': 'bold', 'color': 'green'})
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'link-Pais':
            return html.Div('Contenido de la sección Modelos Numéricos País:',
         style={
             'font-family': 'Open Sans, sans-serif',
             'font-style': 'italic',
             'color': 'green',
             'font-size': '20px',
             'text-shadow': '1px 1px 4px #64946C'  # Sombra de texto
         })
        elif button_id == 'link-Departamentos':
            return html.Div('Contenido de la sección Modelos Numéricos Departamentos:',
         style={
             'font-family': 'Open Sans, sans-serif',
             'font-style': 'italic',
             'color': 'green',
             'font-size': '20px',
             'text-shadow': '1px 1px 4px #64946C'  # Sombra de texto
         })
        elif button_id == 'link-Municipios':
            return html.Div('Contenido de la sección Modelos Numéricos Municipios:',
         style={
             'font-family': 'Open Sans, sans-serif',
             'font-style': 'italic',
             'color': 'green',
             'font-size': '20px',
             'text-shadow': '1px 1px 4px #64946C'  # Sombra de texto
         })
        elif button_id == 'link-Cuencas':
            return html.Div('Contenido de la sección Modelos Numéricos Cuencas Hidrográficas:',
         style={
             'font-family': 'Open Sans, sans-serif',
             'font-style': 'italic',
             'color': 'green',
             'font-size': '20px',
             'text-shadow': '1px 1px 4px #64946C'  # Sombra de texto
         })
        elif button_id == 'link-Subcuencas':
            return html.Div('Contenido de la sección Modelos Numéricos Subcuencas Hidrográficas:',
         style={
             'font-family': 'Open Sans, sans-serif',
             'font-style': 'italic',
             'color': 'green',
             'font-size': '20px',
             'text-shadow': '1px 1px 4px #64946C'  # Sombra de texto
         })
        else:
            return html.Div('Sección no encontrada.', style={'font-weight': 'bold', 'color': 'red', 'font-size': '25px'})
        # Añade aquí más condiciones para otros botones
# Callback para actualizar el contenido principal
@app.callback(
    Output('main-content', 'children'),
    [Input('url', 'pathname')],
)
def display_page(pathname):
    if pathname == '/departamento_WRF_atlantida':
        return html.Div([
            dbc.Badge([
                html.Em("WRF-HN ClimaScope", style={'fontSize': '17px'}),
                ": ",
                "Modelos Numéricos del WRF (Weather Research and Forecasting)"
                ], className="d-flex justify-content-center", color="dark", style={'textAlign': 'center', 'fontSize': '16px'}),
                        html.Hr(),
            dbc.Badge("Datos de Precipitación del departemento de Atlántida por Fecha del Pronóstico 5 días del WRF", className="ms-1", color="success", style={'fontSize': '14px'}),
                        html.Br(),
                        dbc.Row([
                        dash_table.DataTable(
                            id='table-grafica_Atlantida',
                            columns=[{"name": i, "id": i} for i in df_Atlantida.columns],
                            data=df_Atlantida.to_dict('records'),
                            style_table={'height': '260px', 'width': '1325px', 'overflowY': 'auto'},
                            page_size=8,
                            style_data_conditional=[
                                {
                                    'if': {
                                        'row_index': 1,  # Para la primera fila
                                        },
                                    'backgroundColor': 'darkgreen',
                                    'color': 'white'
                                    },
                                {
                                    'if': {
                                        'row_index': 2,  # Para la segunda fila
                                        },
                                    'backgroundColor': 'darkgreen',
                                    'color': 'white'
                                    },
                                {
                                    'if': {
                                        'row_index': 3,  # Para la segunda fila
                                        },
                                    'backgroundColor': 'darkgreen',
                                    'color': 'white'
                                    },
                                {
                                    'if': {
                                        'row_index': 4,  # Para la segunda fila
                                        },
                                    'backgroundColor': 'darkgreen',
                                    'color': 'white'
                                    },
                                {
                                    'if': {
                                        'row_index': 5,  # Para la segunda fila
                                        },
                                    'backgroundColor': 'darkgreen',
                                    'color': 'white'
                                    },
                                {
                                    'if': {
                                        'row_index': 6,  # Para la segunda fila
                                        },
                                    'backgroundColor': '#3B9F38',
                                    'color': 'white'
                                    },
                                # Estilo para celdas seleccionadas
                                {
                                    'if': {'state': 'selected'},  # Aplica el estilo a las celdas seleccionadas
                                    'backgroundColor': 'yellow',
                                    'color': 'black'
                                }
                                ],
                            #row_selectable='single'
                        ),
                        ]),
                        html.Hr(),
                        dbc.Badge("Graficas de Precipitación del Departamento de Atlántida por Fecha del Pronóstico 5 días del WRF", className="ms-1", color="success", style={'fontSize': '14px'}),
                        html.Br(),
                        dbc.Row([
                        dcc.Graph(id='grafica_Atlantida', figure=figAtlanPP, style={'display': 'inline'}),
                        ]),
                        html.Hr(),
                        dbc.Badge("Graficas de Precipitación Promedio del Departamento de Atlántida por Fecha del Pronóstico 5 días del WRF", className="ms-1", color="success", style={'fontSize': '14px'}), # Nueva sección para la gráfica promedios
                        html.Br(),
                        dbc.Row([
                        dcc.Graph(id='grafica_Atlantida_adicional', figure=figAavgT, style={'display': 'inline-block'}),
                        ]),
                        html.Hr(),
                        dbc.Badge("Mapas de Precipitación del Departamento de Atlántida Pronóstico del WRF", className="ms-1", color="success", style={'fontSize': '14px'}),
                        html.Br(),
                        dbc.Row([
                        dcc.Graph(id='mapa_Atlantidaac', figure=fig_map_Atlantidaavg, style={'display': 'inline-block'}),
                        ]),
                        html.Hr(),
                        dbc.Row([
                        dcc.Graph(id='mapa_Atlantidaavg', figure=fig_map_Atlantidaac, style={'display': 'inline-block'}),
                        ]),
                        html.Hr(),
                        dbc.Row([
                        dcc.Graph(id='mapa_Atlantidamin', figure=fig_map_Atlantidamin, style={'display': 'inline-block'}),
                        ]),
                        html.Hr(),
                        dbc.Row([
                        dcc.Graph(id='mapa_Atlantidamax', figure=fig_map_Atlantidamax, style={'display': 'inline-block'}),
                        ]),
                        html.Hr(),
                        dbc.Badge("Visualizador 3D de la Precipitación Promedio del Departamento de Atlántida Pronóstico del WRF", className="ms-1", color="success", style={'fontSize': '14px'}),
                        html.Br(),
                        dbc.Row([
                        dcc.Graph(id='mapa_Atlantidaavg3d', figure=fig_map_Atlantidaavg3d, style={'display': 'inline-block'}),
                        ]),
                        html.Hr(),
                        dbc.Badge("Visualizador 3D de la Precipitación Acumulada del Departamento de Atlántida Pronóstico del WRF", className="ms-1", color="success", style={'fontSize': '14px'}),
                        html.Br(),
                        dbc.Row([
                        dcc.Graph(id='mapa_Atlantidaac3d', figure=fig_map_Atlantidaac3d, style={'display': 'inline-block'}),
                        ]),
                        html.Hr(),
                        dbc.Badge("Visualizador 3D de la Precipitación Mínima del Departamento de Atlántida Pronóstico del WRF", className="ms-1", color="success", style={'fontSize': '14px'}),
                        html.Br(),
                        dbc.Row([
                        dcc.Graph(id='mapa_Atlantidamin3d', figure=fig_map_Atlantidamin3d, style={'display': 'inline-block'}),
                        ]),
                        html.Hr(),
                        dbc.Badge("Visualizador 3D de la Precipitación Máxima del Departamento de Atlántida Pronóstico del WRF", className="ms-1", color="success", style={'fontSize': '14px'}),
                        html.Br(),
                        dbc.Row([
                        dcc.Graph(id='mapa_Atlantidamax3d', figure=fig_map_Atlantidamax3d, style={'display': 'inline-block'}),
                        ]),
                        html.Hr(),
            # ... Contenido específico para Atlántida ...
        ])
    elif pathname == '/WRF_pais_Honduras':
        return html.Div([
            html.Hr(),
            dbc.Badge("Datos de Precipitación de Honduras Pronóstico del WR", className="ms-6"),
                        html.Br(),
                        dash_table.DataTable(
                            id='table-grafica_Atlantida',
                            columns=[{"name": i, "id": i} for i in df_Atlantida.columns],
                            data=df_Atlantida.to_dict('records'),
                            style_table={'height': '5px', 'width': '1350px', 'overflowY': 'auto'},
                            page_size=6,
                            style_data_conditional=[
                                {
                                    'if': {
                                        'row_index': 1,  # Para la primera fila
                                        },
                                    'backgroundColor': 'green',
                                    'color': 'white'
                                    },
                                {
                                    'if': {
                                        'row_index': 2,  # Para la segunda fila
                                        },
                                    'backgroundColor': 'green',
                                    'color': 'white'
                                    },
                                {
                                    'if': {
                                        'row_index': 3,  # Para la segunda fila
                                        },
                                    'backgroundColor': 'green',
                                    'color': 'white'
                                    },
                                {
                                    'if': {
                                        'row_index': 4,  # Para la segunda fila
                                        },
                                    'backgroundColor': 'green',
                                    'color': 'white'
                                    },
                                {
                                    'if': {
                                        'row_index': 5,  # Para la segunda fila
                                        },
                                    'backgroundColor': 'green',
                                    'color': 'white'
                                    }
                                ],
                            #row_selectable='multi'
                        ),
                        html.Hr(),
                        dbc.Badge("Graficas de Precipitación del departamento de Atlántida", className="ms-6"),
                        html.Br(),
                        dcc.Graph(id='grafica_Atlantida', figure=figAtlanPP, style={'display': 'inline-block'}),
                        html.Hr(),
                        dbc.Badge("Graficas de Precipitación promedio del departamento de Atlántida Pronostico WRF", className="ms-6"), # Nueva sección para la gráfica promedios
                        html.Br(),
                        dcc.Graph(id='grafica_Atlantida_adicional', figure=figAavgT, style={'display': 'inline-block'}),
                        html.Hr(),
                        dbc.Badge("Mapas de Precipitación del departamento de Atlántida", className="ms-6"),
                        html.Br(),
                        dcc.Graph(id='mapa_Atlantidaac', figure=fig_map_Atlantidaac, style={'display': 'inline-block'}),
                        dcc.Graph(id='mapa_Atlantidaavg', figure=fig_map_Atlantidaavg, style={'display': 'inline-block'}),
                        html.Br(),
                        dcc.Graph(id='mapa_Atlantidamin', figure=fig_map_Atlantidamin, style={'display': 'inline-block'}),
                        dcc.Graph(id='mapa_Atlantidamax', figure=fig_map_Atlantidamax, style={'display': 'inline-block'}),
                        html.Br(),
                        html.Hr(),
                        dbc.Badge("Visualizador 3D de la Precipitación del departamento de Atlántida", className="ms-6"),
                        html.Br(),
                        dcc.Graph(id='mapa_Atlantidaavg3d', figure=fig_map_Atlantidaavg3d, style={'display': 'inline-block'}),
                        html.Hr(),
        ])
    # ... Agrega más condiciones para otros elementos de la barra de navegación ...
    elif pathname == '/WRF_HN_ClimaScope':
        return html.Div([dbc.Alert([
            html.H4("Bienvenido al Visualizador de Datos Pronóstico del WRF", className="alert-heading"),
            html.P(["Te encuentras en la sección inicial de la Analitical Web Aplication ", html.Strong(html.Em("WRF-HN ClimaScope", style={'color': '#427122'}))]),
            html.P([
                "La aplicación está enfocada para visualizar datos meteorológicos y geográficos pronosticados por el modelo WRF ",
                html.A(html.Strong("(Weather Research and Forecasting)"), href="https://www.mmm.ucar.edu/weather-research-and-forecasting-model", target="_blank", style={'color': '#427122'}, className="custom-link"),
                " para diferentes regiones de Honduras."
            ]),
            #html.Br(),  # Salto de línea aquí
            html.Hr(),
            html.P(["Arriba tiene un menú de navegación que te permitirá obtener información del pronóstico del modelo WRF para las diferentes regiones de Honduras, los modelos numéricos están organizados por País, Departamento, Municipios, Cuencas Hidrográficas y Subcuencas Hidrográficas."]),
        ], dismissable=True)
    ], className='page-container')
    else:
        return dbc.Row([
            html.Div('ERROR 404 / Sección no encontrada.', style={'textAlign': 'center', 'font-weight': 'bold', 'color': 'red', 'font-size': '50px'})
        ])


@app.callback(
    [Output("spinner-output", "children"),
    Output("spinner-wrapper", "style")],  # Actualizará el contenido del div envuelto por el spinner
    [Input('link-ClimaScope', 'n_clicks'),
    Input('link-Pais', 'n_clicks'),
    Input('link-Departamentos', 'n_clicks'),
    Input('link-Municipios', 'n_clicks'),
    Input('link-Cuencas', 'n_clicks'),
    Input('link-Subcuencas', 'n_clicks')]
)
def update_output(link_INICIO_n_clicks, link_Pais_n_clicks, link_Departamentoz_n_clicks, link_Municipios_n_clicks, link_Cuencas_n_clicks, link_Subcuencas_n_clicks):

    # Puedes usar dash.callback_context para identificar qué botón fue presionado
    ctx = dash.callback_context
    if not ctx.triggered:
        return "",  {"display": "none"}  # Ocultar el spinner si no se ha hecho clic  # No se ha presionado ningún botón
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        # Simula un procesamiento que toma tiempo
        time.sleep(15)  # Ajusta este tiempo según tus necesidades
    processed_content = f"Procesado: {button_id}"
    return processed_content, {"display": "none"}  # Mostrar contenido procesado y ocultar el spinner
# Callback para actualizar el contenido del spinner y controlar su visibilidad

@app.callback(
    Output('current-year', 'children'),
    Input('url', 'pathname')  # Se activa con cualquier cambio en la URL, o puedes usar otro Input
)
def update_footer(pathname):
    return str(datetime.now().year)

# Ejecutar la app
if __name__ == '__main__':
    app.run(debug=True)
