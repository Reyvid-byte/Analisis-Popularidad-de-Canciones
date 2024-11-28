import dash
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# Cargar el modelo previamente entrenado
with open('modelo_music_predictor.pkl', 'rb') as f:
    model = pickle.load(f)

# Datos ficticios para simular y_test, y_pred, y características importantes
# En un caso real, estos datos se generarán durante el entrenamiento y validación del modelo
y_test = [50, 60, 40, 80, 90, 55, 65]
y_pred = [48, 62, 38, 82, 88, 53, 67]

# Importancia de características (ficticia para demo; en un caso real, usarías rf_model.feature_importances_)
features = ['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 
            'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
feature_importances = [0.15, 0.12, 0.10, 0.20, 0.05, 0.08, 0.10, 0.05, 0.10, 0.05]

# Crear un DataFrame para ordenar las importancias
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Crear el dashboard
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Predicción de Popularidad de Canciones", style={
        "textAlign": "center", 
        "color": "#2c3e50", 
        "backgroundColor": "#ecf0f1", 
        "padding": "10px", 
        "borderRadius": "5px"}),

    # Inputs de características
    html.Div(
        style={"margin": "20px", "padding": "20px", "backgroundColor": "#f8f9fa", "borderRadius": "10px"},
        children=[
            html.Label("Introduce las características musicales:", style={"color": "#34495e", "fontSize": "18px"}),
            html.Div(
                style={"display": "flex", "flexWrap": "wrap", "gap": "10px"},
                children=[
                    dcc.Input(id="acousticness-input", type="number", placeholder="Acousticness", style={"flex": "1", "minWidth": "150px"}),
                    dcc.Input(id="danceability-input", type="number", placeholder="Danceability", style={"flex": "1", "minWidth": "150px"}),
                    dcc.Input(id="duration_ms-input", type="number", placeholder="Duration (ms)", style={"flex": "1", "minWidth": "150px"}),
                    dcc.Input(id="energy-input", type="number", placeholder="Energy", style={"flex": "1", "minWidth": "150px"}),
                    dcc.Input(id="instrumentalness-input", type="number", placeholder="Instrumentalness", style={"flex": "1", "minWidth": "150px"}),
                    dcc.Input(id="liveness-input", type="number", placeholder="Liveness", style={"flex": "1", "minWidth": "150px"}),
                    dcc.Input(id="loudness-input", type="number", placeholder="Loudness", style={"flex": "1", "minWidth": "150px"}),
                    dcc.Input(id="speechiness-input", type="number", placeholder="Speechiness", style={"flex": "1", "minWidth": "150px"}),
                    dcc.Input(id="tempo-input", type="number", placeholder="Tempo", style={"flex": "1", "minWidth": "150px"}),
                    dcc.Input(id="valence-input", type="number", placeholder="Valence", style={"flex": "1", "minWidth": "150px"}),
                ]
            ),
            html.Div([
                html.Label("Selecciona la clave (Key):", style={"color": "#34495e", "fontSize": "16px"}),
                dcc.Dropdown(
                    id="key-input",
                    options=[{"label": key, "value": key} for key in ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']],
                    value='C',
                    style={"width": "200px"}
                ),
            ], style={"marginTop": "10px"}),

            html.Div([
                html.Label("Selecciona el modo (Mode):", style={"color": "#34495e", "fontSize": "16px"}),
                dcc.Dropdown(
                    id="mode-input",
                    options=[{"label": mode, "value": mode} for mode in ['Major', 'Minor']],
                    value='Major',
                    style={"width": "200px"}
                ),
            ], style={"marginTop": "10px"}),

            html.Div([
                html.Label("Selecciona el género musical:", style={"color": "#34495e", "fontSize": "16px"}),
                dcc.Dropdown(
                    id="music_genre-input",
                    options=[{"label": genre, "value": genre} for genre in ['Alternative', 'Anime', 'Blues', 'Classical', 'Country', 
                                                                            'Electronic', 'Hip-Hop', 'Jazz', 'Rap', 'Rock']],
                    value='Alternative',
                    style={"width": "200px"}
                ),
            ], style={"marginTop": "10px"}),
        ]
    ),

    html.Div([
        html.Button("Predecir Popularidad", id="predict-button", style={
            "backgroundColor": "#2ecc71", 
            "color": "white", 
            "border": "none", 
            "padding": "10px 20px", 
            "borderRadius": "5px", 
            "cursor": "pointer"}),
        html.Div(id="prediction-output", style={"textAlign": "center", "color": "#34495e", "fontSize": "20px", "marginTop": "20px"}),
    ], style={"textAlign": "center", "marginTop": "20px"}),

    # Gráficas
    html.Div([
        dcc.Graph(id="sorted-predictions-graph", style={"marginTop": "20px"}),
        dcc.Graph(id="feature-importance-graph", style={"marginTop": "20px"}),
    ]),
])


# Callback para predecir y graficar
@app.callback(
    [Output("prediction-output", "children"),
     Output("sorted-predictions-graph", "figure"),
     Output("feature-importance-graph", "figure")],
    [Input("acousticness-input", "value"),
     Input("danceability-input", "value"),
     Input("duration_ms-input", "value"),
     Input("energy-input", "value"),
     Input("instrumentalness-input", "value"),
     Input("liveness-input", "value"),
     Input("loudness-input", "value"),
     Input("speechiness-input", "value"),
     Input("tempo-input", "value"),
     Input("valence-input", "value"),
     Input("key-input", "value"),
     Input("mode-input", "value"),
     Input("music_genre-input", "value")]
)
def update_dashboard(acousticness, danceability, duration_ms, energy, instrumentalness, liveness, loudness, speechiness, tempo, valence, key, mode, music_genre):
    # Verificar que todos los valores estén presentes
    if None in [acousticness, danceability, duration_ms, energy, instrumentalness, liveness, loudness, speechiness, tempo, valence, key, mode, music_genre]:
        return "Por favor, ingresa todos los valores para hacer una predicción.", {}, {}

    # Crear un diccionario con las entradas del usuario
    example = {
        'acousticness': acousticness,
        'danceability': danceability,
        'duration_ms': duration_ms,
        'energy': energy,
        'instrumentalness': instrumentalness,
        'liveness': liveness,
        'loudness': loudness,
        'speechiness': speechiness,
        'tempo': tempo,
        'valence': valence,
        f'key_{key}': 1,
        f'mode_{mode}': 1,
        f'music_genre_{music_genre}': 1,
    }

    # Crear DataFrame a partir del diccionario
    df_example = pd.DataFrame([example])

    # Crear todas las columnas posibles (las que el modelo espera)
    expected_columns = model.feature_names_in_
    for col in expected_columns:
        if col not in df_example.columns:
            df_example[col] = 0  # Agregar columnas faltantes con valor 0

    # Ordenar las columnas de acuerdo al modelo
    df_example = df_example[expected_columns]

    # Realizar la predicción
    prediction = model.predict(df_example)[0]

    # Graficar las predicciones ordenadas vs valores reales
    sorted_indices = np.argsort(y_pred)
    sorted_real = np.array(y_test)[sorted_indices]
    sorted_pred = np.array(y_pred)[sorted_indices]

    fig_sorted = go.Figure()
    fig_sorted.add_trace(go.Scatter(y=sorted_real, mode='lines+markers', name='Valores Reales'))
    fig_sorted.add_trace(go.Scatter(y=sorted_pred, mode='lines+markers', name='Predicciones'))
    fig_sorted.update_layout(
        title="Predicciones Ordenadas vs Valores Reales",
        xaxis_title="Índice Ordenado",
        yaxis_title="Popularidad",
        legend_title="Leyenda",
    )

    # Graficar la importancia de características
    fig_importance = go.Figure(
        go.Bar(
            x=importance_df['Importance'], 
            y=importance_df['Feature'], 
            orientation='h',
            marker_color='skyblue'
        )
    )
    fig_importance.update_layout(
        title="Importancia de las Características",
        xaxis_title="Importancia",
        yaxis_title="Característica",
        yaxis=dict(autorange="reversed"),
    )

    # Mostrar la predicción
    prediction_text = f"Predicción de popularidad: {prediction:.2f}"

    return prediction_text, fig_sorted, fig_importance


# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True, port=8100)
