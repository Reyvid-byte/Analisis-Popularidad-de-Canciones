import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Crear la aplicación Dash
app = dash.Dash(__name__)
app.title = "Análisis Musical"

# Cargar y procesar tus datos
df = pd.read_csv(r"C:\Users\Rey David\OneDrive\Documentos\ProyectoDataBase\DF_Music_Limpio.csv")

# Eliminar la columna 'Unnamed: 0' y canciones sin nombre
df = df.drop(columns=['Unnamed: 0'])
df = df[df['track_name'] != 's/tn']

# Normalizar las columnas relevantes (incluyendo popularidad)
scaler = MinMaxScaler()
columns_to_normalize = ["popularity", "danceability", "energy", "loudness", "valence", "tempo", "instrumentalness"]
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Obtener todas las columnas numéricas para comparar
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
numeric_columns.remove('popularity')  # Excluir 'popularity' de la comparación

# Top 10 Artistas
top_artists = (
    df.groupby(["artist_name", "music_genre"])["popularity"]
    .mean()
    .reset_index()
    .sort_values("popularity", ascending=False)
    .head(10)
)
fig_top_artists = px.bar(
    top_artists,
    x="artist_name",
    y="popularity",
    color="music_genre",
    title="Top 10 Artistas por Popularidad",
    color_discrete_sequence=px.colors.qualitative.Pastel,
)
fig_top_artists.update_layout(
    xaxis_title="Artista",
    yaxis_title="Popularidad Normalizada",
    plot_bgcolor="#ffffff",
    paper_bgcolor="#f9f9f9",
)

# Top 10 Canciones
top_songs = (
    df.groupby(["track_name", "music_genre"])["popularity"]
    .mean()
    .reset_index()
    .sort_values("popularity", ascending=False)
    .head(10)
)
fig_top_songs = px.bar(
    top_songs,
    x="track_name",
    y="popularity",
    color="music_genre",
    title="Top 10 Canciones por Popularidad",
    color_discrete_sequence=px.colors.qualitative.Pastel1,
)
fig_top_songs.update_layout(
    xaxis_title="Canción",
    yaxis_title="Popularidad Normalizada",
    plot_bgcolor="#ffffff",
    paper_bgcolor="#f9f9f9",
)

# Scatter plot: Energy vs Loudness
fig_energy_loudness = px.scatter(
    df,
    x="energy",
    y="loudness",
    color="music_genre",
    title="Energy vs Loudness por Género Musical",
    color_discrete_sequence=px.colors.qualitative.Pastel
)
fig_energy_loudness.update_layout(
    xaxis_title="Energy Normalizada",
    yaxis_title="Loudness Normalizada",
    plot_bgcolor="#ffffff",
    paper_bgcolor="#f9f9f9",
    title_font=dict(size=20, color="#2c3e50"),
)

# Gráfico de Densidad 2D para Energy vs Acousticness
fig_energy_acousticness = px.density_contour(
    df,
    x="energy",
    y="acousticness",
    title="Densidad 2D de Energy vs Acousticness",
)
fig_energy_acousticness.update_layout(
    xaxis_title="Energy Normalizada",
    yaxis_title="Acousticness Normalizada",
    plot_bgcolor="#ffffff",
    paper_bgcolor="#f9f9f9",
    title_font=dict(size=20, color="#2c3e50"),
)

# Diseño del Dashboard
app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "backgroundColor": "#f9f9f9", "padding": "20px"},
    children=[
        html.H1("Dashboard de Análisis Musical", style={"textAlign": "center", "color": "#2c3e50"}),
        html.Div(
            style={"marginBottom": "20px"},
            children=[ 
                html.Label("Selecciona una característica musical:", style={"color": "#34495e"}),
                dcc.Dropdown(
                    id="feature-dropdown",
                    options=[{"label": col.capitalize(), "value": col} for col in numeric_columns],
                    value="danceability",
                    clearable=False,
                    style={"width": "50%", "margin": "0 auto"},
                ),
            ],
        ),
        dcc.Graph(id="dual-bar-chart"),
        html.Div(
            style={"marginTop": "30px"},
            children=[
                html.H2("Top 10 Artistas", style={"textAlign": "center", "color": "#2c3e50"}),
                dcc.Graph(id="top-artists", figure=fig_top_artists),
            ],
        ),
        html.Div(
            style={"marginTop": "30px"},
            children=[
                html.H2("Top 10 Canciones", style={"textAlign": "center", "color": "#2c3e50"}),
                dcc.Graph(id="top-songs", figure=fig_top_songs),
            ],
        ),
        html.Div(
            style={"marginTop": "30px"},
            children=[
                html.H2("Energy vs Loudness (Scatter Plot)", style={"textAlign": "center", "color": "#2c3e50"}),
                dcc.Graph(id="energy-loudness-plot", figure=fig_energy_loudness),
            ],
        ),
        html.Div(
            style={"marginTop": "30px"},
            children=[
                html.H2("Energy vs Acousticness (Densidad 2D)", style={"textAlign": "center", "color": "#2c3e50"}),
                dcc.Graph(id="energy-acousticness-plot", figure=fig_energy_acousticness),
            ],
        ),
    ],
)

# Callback para la gráfica doble de barras
@app.callback(
    dash.dependencies.Output("dual-bar-chart", "figure"),
    [dash.dependencies.Input("feature-dropdown", "value")],
)
def update_dual_bar_chart(selected_feature):
    df_grouped = df.groupby("music_genre").agg(
        {"popularity": "mean", selected_feature: "mean"}
    ).reset_index()
    fig = px.bar(
        df_grouped,
        x="music_genre",
        y=["popularity", selected_feature],
        barmode="group",
        title=f"Popularidad vs {selected_feature.capitalize()} por Género Musical (Normalizado)",
        color_discrete_sequence=["#1f77b4", "#ff7f0e"],
    )
    fig.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#f9f9f9",
        title_font=dict(size=20, color="#2c3e50"),
        xaxis_title="Género Musical",
        yaxis_title="Valores Normalizados",
        legend_title="Métrica",
    )
    return fig

# Ejecutar el servidor
if __name__ == "__main__":
    app.run_server(debug=True, port=8081)
