import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(page_title="Regresión Logística", layout="wide")

# Inyección de CSS personalizado
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Crear archivo CSS directamente desde el código
css_content = """
/* Global Styles */
body {
    font-family: 'Roboto', sans-serif;
    font-size: 16px;
    background-color: #f8f9fa;
    color: #333333;
}

/* Encabezados */
h1 {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    color: #4CAF50;
    margin-bottom: 20px;
}

h2 {
    font-size: 2rem;
    color: #2c3e50;
    margin-bottom: 15px;
}

h3 {
    font-size: 1.5rem;
    color: #333333;
    margin-bottom: 10px;
}

/* Barra lateral */
.sidebar .sidebar-content {
    background-color: #2c3e50;
    color: #ecf0f1;
}

.sidebar .stRadio > label {
    font-size: 1rem;
    color: #ecf0f1;
    font-weight: bold;
    margin: 10px 0;
}

.sidebar .stRadio .st-bd {
    background-color: #4CAF50;
    color: white;
    padding: 8px 10px;
    border-radius: 5px;
    transition: transform 0.3s ease-in-out;
}

.sidebar .stRadio .st-bd:hover {
    transform: scale(1.1);
}

/* Botones */
.stButton button {
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 16px;
    cursor: pointer;
    margin: 10px;
    transition: background-color 0.3s ease, transform 0.2s ease;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.stButton button:hover {
    background-color: #45a049;
    transform: translateY(-2px);
}

/* Centrado de botones */
.button-container {
    display: flex;
    justify-content: center;
    gap: 10px;
    margin: 20px 0;
}

/* Tablas */
table {
    border-collapse: collapse;
    width: 100%;
    margin-top: 20px;
}

table, th, td {
    border: 1px solid #ddd;
    text-align: left;
    padding: 10px;
}

th {
    background-color: #4CAF50;
    color: white;
}

tr:nth-child(even) {
    background-color: #f2f2f2;
}

tr:hover {
    background-color: #ddd;
}

/* Gráficos */
.plot-container {
    margin: 20px 0;
    max-width: 100%;
    padding: 15px;
    background-color: #ffffff;
    border: 1px solid #ddd;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

/* Transiciones en gráficos */
.plot-container:hover {
    transform: scale(1.02);
    transition: transform 0.3s ease-in-out;
}

/* Animaciones suaves */
button, input, .plot-container {
    transition: all 0.3s ease-in-out;
}

/* Diseño Responsivo */
@media (max-width: 768px) {
    body {
        font-size: 14px;
    }

    h1 {
        font-size: 2rem;
    }

    h2 {
        font-size: 1.5rem;
    }

    .sidebar .sidebar-content {
        font-size: 14px;
    }
}

"""

# Guardar el contenido CSS en un archivo temporal
css_file = "styles.css"
with open(css_file, "w") as file:
    file.write(css_content)

# Inyectar el CSS
local_css(css_file)

# Título principal
st.title("🔍 Regresión Logística")

# Función para inicializar variables en session_state
def init_session_state():
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'X' not in st.session_state:
        st.session_state.X = None
    if 'y' not in st.session_state:
        st.session_state.y = None
    if 'feature_variables' not in st.session_state:
        st.session_state.feature_variables = []
    if 'target_variable' not in st.session_state:
        st.session_state.target_variable = None
    if 'X_train_scaled' not in st.session_state:
        st.session_state.X_train_scaled = None
    if 'X_test_scaled' not in st.session_state:
        st.session_state.X_test_scaled = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None

# Inicializar variables de estado
init_session_state()

# Sidebar para cargar el archivo y navegar entre pasos
st.sidebar.title("Navegación")

# Paso de navegación
st.sidebar.markdown("""
    <style>
        .css-1v3fvcr > div:first-child {
            background-color: #f0f8ff;
            color: #007acc;
            border-radius: 10px;
            padding: 10px;
        }
        .css-1v3fvcr > div:first-child:hover {
            background-color: #dbefff;
        }
    </style>
""", unsafe_allow_html=True)

step = st.sidebar.radio("Selecciona el paso:", [
    "📂 Paso 1: Cargar Datos",
    "📊 Paso 2: Seleccionar Variables",
    "✂️ Paso 3: Dividir Datos",
    "🤖 Paso 4: Entrenar Modelo",
    "📈 Paso 5: Evaluar Modelo",
    "🔮 Paso 6: Predicción con Nuevos Datos"
])


# Paso 1: Cargar y visualizar los datos
if step == "📂 Paso 1: Cargar Datos":
    st.markdown("""
    <style>
        .welcome-text {
            text-align: center; /* Centra el texto */
            font-family: 'Georgia', serif; /* Cambia la fuente a una más elegante */
            font-size: 20px; /* Ajusta el tamaño de la letra */
            color: black; /* Cambia el color a negro */
            font-weight: bold; /* Puedes usar bold para hacerlo más grueso */
            margin-top: 20px; /* Espaciado superior */
            margin-bottom: 20px; /* Espaciado inferior */
        }
    </style>
    <p class="welcome-text">
        🧠¡Tu viaje hacia el análisis predictivo comienza aquí! 🔍
    </p>
""", unsafe_allow_html=True)
    st.header("📂 Paso 1: Cargar Datos")

    # Sidebar para cargar el archivo
    st.sidebar.subheader("Carga de Datos")
    uploaded_file = st.sidebar.file_uploader("Sube un archivo de datos (CSV o XLSX)", type=["csv", "xlsx"])

    if uploaded_file:
        # Leer el archivo según su extensión
        try:
            if uploaded_file.name.endswith('.csv'):
                st.session_state.df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                st.session_state.df = pd.read_excel(uploaded_file)


            # Mostrar una vista previa de los datos cargados
            st.write("### Vista previa de los datos cargados:")
            st.dataframe(st.session_state.df.head())

            # Mostrar información básica de los datos
            st.write("### Información de los datos:")
            st.write(st.session_state.df.describe())

            # Matriz de correlación con Plotly
            st.write("### Matriz de Correlación")
            correlation_matrix = st.session_state.df.corr()  # Calcular la matriz de correlación
            fig_corr = px.imshow(
                correlation_matrix,
                text_auto=".2f",  # Mostrar los valores con 2 decimales
                color_continuous_scale="RdBu_r",  # Escala de colores
                labels=dict(color="Correlación"),
                title="Matriz de Correlación"
            )
            fig_corr.update_layout(
                title_font_size=20,
                title_x=0.5,
                xaxis_title="Variables",
                yaxis_title="Variables",
                coloraxis_showscale=True,
                width=800,  # Ajustar el ancho del gráfico
                height=600  # Ajustar la altura del gráfico
            )
            st.plotly_chart(fig_corr, use_container_width=False)
        except Exception as e:
            st.error(f"Ocurrió un error al leer el archivo: {e}")
    else:
        st.info("Por favor, carga un archivo CSV o XLSX para comenzar.")

# Paso 2: Seleccionar variables
elif step == "📊 Paso 2: Seleccionar Variables":
    st.header("📊 Paso 2: Seleccionar Variables")

    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ Por favor, ve al Paso 1 y carga los datos primero.")
    else:
        # Seleccionar variables independientes y dependiente
        st.sidebar.subheader("Seleccionar Variables")
        all_columns = st.session_state.df.columns.tolist()

        # Seleccionar la variable dependiente
        st.session_state.target_variable = st.sidebar.selectbox(
            "Selecciona la variable dependiente (Y):", 
            all_columns, 
            help="Elige la columna que quieres predecir"
        )

        # Seleccionar las variables independientes
        st.session_state.feature_variables = st.sidebar.multiselect(
            "Selecciona las variables independientes (X):", 
            [col for col in all_columns if col != st.session_state.target_variable],
            help="Elige una o más columnas para entrenar el modelo"
        )

        # Validación de selección
        if st.session_state.target_variable and st.session_state.feature_variables:
            # Guardar las variables seleccionadas
            st.session_state.X = st.session_state.df[st.session_state.feature_variables]
            st.session_state.y = st.session_state.df[st.session_state.target_variable]

            # Mostrar las variables seleccionadas
            st.markdown("### ✅ Resumen de Selección")
            st.success(f"**Variable dependiente (Y):** {st.session_state.target_variable}")
            st.info(f"**Variables independientes (X):** {', '.join(st.session_state.feature_variables)}")

            # Visualización: Distribución de variables independientes
            st.markdown("### 📊 Distribución de las Variables Independientes")
            col1, col2 = st.columns(2)
            for i, feature in enumerate(st.session_state.feature_variables):
                with col1 if i % 2 == 0 else col2:
                    fig_feat = px.histogram(
                        st.session_state.df, 
                        x=feature, 
                        nbins=30, 
                        title=f"Distribución de {feature}",
                        color_discrete_sequence=["#007acc"]
                    )
                    fig_feat.update_layout(
                        title_font_size=14, 
                        title_x=0.5,
                        margin=dict(t=40, b=30)
                    )
                    st.plotly_chart(fig_feat, use_container_width=True)

            # Visualización: Distribución de la variable dependiente
            st.markdown("### 🎯 Distribución de la Variable Dependiente")
            fig_target = px.histogram(
                st.session_state.df, 
                x=st.session_state.target_variable, 
                title=f"Distribución de {st.session_state.target_variable}",
                color_discrete_sequence=["#4CAF50"]
            )
            fig_target.update_layout(
                title_font_size=16, 
                title_x=0.5,
                margin=dict(t=50, b=30)
            )
            st.plotly_chart(fig_target, use_container_width=True)
        else:
            st.warning("⚠️ Por favor, selecciona la variable dependiente y al menos una variable independiente.")


# Paso 3: Dividir Datos
elif step == "✂️ Paso 3: Dividir Datos":
    st.header("✂️ Paso 3: Dividir Datos")

    if "X" not in st.session_state or st.session_state.X is None or "y" not in st.session_state or st.session_state.y is None:
        st.warning("⚠️ Por favor, completa los pasos anteriores antes de continuar.")
    else:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        # Parámetros de división de datos
        st.sidebar.subheader("⚙️ Parámetros de División de Datos")
        test_size = st.sidebar.slider(
            "Tamaño del conjunto de prueba:", 
            0.1, 0.5, 0.3, 
            step=0.05, 
            help="Porcentaje de datos utilizados para pruebas (ejemplo: 0.3 = 30%)"
        )
        random_state = st.sidebar.number_input(
            "Semilla de aleatoriedad (random_state):", 
            value=42, 
            step=1, 
            help="Usa un valor fijo para resultados reproducibles"
        )

        # Dividir los datos
        st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = train_test_split(
            st.session_state.X, st.session_state.y, test_size=test_size, random_state=random_state
        )

        # Mostrar tamaños de los conjuntos
        st.markdown("### ✅ Resumen de la División de Datos")
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**Conjunto de entrenamiento:** {st.session_state.X_train.shape[0]} muestras")
        with col2:
            st.info(f"**Conjunto de prueba:** {st.session_state.X_test.shape[0]} muestras")

        # Escalado de datos
        st.markdown("### 🔄 Escalado de Datos")
        st.session_state.scaler = StandardScaler()
        st.session_state.X_train_scaled = st.session_state.scaler.fit_transform(st.session_state.X_train)
        st.session_state.X_test_scaled = st.session_state.scaler.transform(st.session_state.X_test)

        st.success("🔑 Los datos han sido divididos y escalados correctamente.")

        # Visualización de las distribuciones antes y después del escalado
        st.markdown("### 📊 Comparación de Escalado de Datos")
        col1, col2 = st.columns(2)

        # Distribución original
        with col1:
            st.markdown("#### Datos Originales")
            for i, feature in enumerate(st.session_state.feature_variables[:2]):  # Mostrar hasta 2 variables
                fig_original = px.histogram(
                    st.session_state.X_train, 
                    x=feature, 
                    nbins=30, 
                    title=f"Distribución de {feature} (Original)"
                )
                fig_original.update_layout(
                    title_font_size=14, 
                    title_x=0.5, 
                    margin=dict(t=40, b=30)
                )
                st.plotly_chart(fig_original, use_container_width=True)

        # Distribución escalada
        with col2:
            st.markdown("#### Datos Escalados")
            for i, feature in enumerate(st.session_state.feature_variables[:2]):  # Mostrar hasta 2 variables
                fig_scaled = px.histogram(
                    pd.DataFrame(st.session_state.X_train_scaled, columns=st.session_state.feature_variables),
                    x=feature, 
                    nbins=30, 
                    title=f"Distribución de {feature} (Escalado)",
                    color_discrete_sequence=["#4CAF50"]
                )
                fig_scaled.update_layout(
                    title_font_size=14, 
                    title_x=0.5, 
                    margin=dict(t=40, b=30)
                )
                st.plotly_chart(fig_scaled, use_container_width=True)


# Paso 4: Entrenar Modelo
elif step == "🤖 Paso 4: Entrenar Modelo":
    st.header("🤖 Paso 4: Entrenar Modelo")

    if "X_train_scaled" not in st.session_state or st.session_state.X_train_scaled is None:
        st.warning("⚠️ Por favor, completa los pasos anteriores antes de continuar.")
    else:
        from sklearn.linear_model import LogisticRegression

        # Parámetros del modelo
        st.sidebar.subheader("🔧 Parámetros del Modelo")
        C_value = st.sidebar.number_input(
            "Valor de regularización (C):", 
            value=1.0, 
            help="Controla la regularización. Valores más bajos incrementan la regularización."
        )
        max_iter = st.sidebar.number_input(
            "Número máximo de iteraciones:", 
            value=100, step=10, 
            help="Número de iteraciones máximas para la convergencia del modelo."
        )

        # Entrenamiento del modelo
        st.session_state.model = LogisticRegression(C=C_value, max_iter=max_iter)
        st.session_state.model.fit(st.session_state.X_train_scaled, st.session_state.y_train)

        st.success("✅ El modelo ha sido entrenado exitosamente.")

        # Mostrar coeficientes del modelo
        st.markdown("### 📋 Coeficientes del Modelo")
        coef_df = pd.DataFrame({
            'Variable': st.session_state.feature_variables,
            'Coeficiente': st.session_state.model.coef_[0]
        }).sort_values(by='Coeficiente', ascending=False)

        # Estilo de la tabla
        st.write("#### Tabla de Coeficientes:")
        st.dataframe(coef_df.style.format({"Coeficiente": "{:.4f}"}).background_gradient(cmap="coolwarm"))

        # Visualización de los coeficientes
        st.markdown("### 📊 Visualización de los Coeficientes")
        fig_coef = px.bar(
            coef_df, 
            x='Variable', 
            y='Coeficiente', 
            title='Coeficientes del Modelo', 
            color='Coeficiente',
            color_continuous_scale="RdBu_r",
            labels={'Coeficiente': 'Valor del Coeficiente', 'Variable': 'Variables'}
        )
        fig_coef.update_layout(
            title_font_size=16,
            title_x=0.5,
            margin=dict(t=50, b=30),
            xaxis_title="Variables",
            yaxis_title="Valor del Coeficiente",
            coloraxis_showscale=True
        )
        st.plotly_chart(fig_coef, use_container_width=True)


# Paso 5: Evaluar Modelo
elif step == "📈 Paso 5: Evaluar Modelo":
    st.header("📈 Paso 5: Evaluar Modelo")

    if "model" not in st.session_state or st.session_state.model is None:
        st.warning("⚠️ Por favor, entrena el modelo en el Paso 4 antes de continuar.")
    else:
        # Evaluación del modelo
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
        from sklearn.model_selection import cross_val_score

        # Predicciones y probabilidades
        y_pred = st.session_state.model.predict(st.session_state.X_test_scaled)
        y_pred_proba = st.session_state.model.predict_proba(st.session_state.X_test_scaled)[:, 1]

        # Métricas de evaluación
        st.markdown("### ✅ Métricas de Evaluación")
        accuracy = accuracy_score(st.session_state.y_test, y_pred)
        st.success(f"**Exactitud (Accuracy):** {accuracy:.2f}")

        st.markdown("#### 📋 Reporte de Clasificación")
        report = classification_report(st.session_state.y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(
            report_df.style.format(
                subset=["precision", "recall", "f1-score"], 
                formatter="{:.2f}"
            ).background_gradient(cmap="coolwarm")
        )

        # Matriz de Confusión
        st.markdown("### 🔢 Matriz de Confusión")
        cm = confusion_matrix(st.session_state.y_test, y_pred)
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Predicción", y="Verdadero", color="Frecuencia"),
            title="Matriz de Confusión"
        )
        fig_cm.update_layout(
            xaxis_title="Predicción",
            yaxis_title="Verdadero",
            coloraxis_showscale=True
        )
        st.plotly_chart(fig_cm, use_container_width=True)

       # Curva ROC y AUC - Alternativa
        st.markdown("### 🎯 Curva ROC (Alternativa)")
        fpr, tpr, _ = roc_curve(st.session_state.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Crear la curva ROC con sombreado suave
        fig_roc_alt = go.Figure()

        # Área sombreada bajo la curva
        fig_roc_alt.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", 
            line=dict(width=0.5, color="rgba(0, 100, 255, 0.2)"),
            fill='tozeroy', name="AUC Área", hoverinfo='skip'
        ))

        # Línea ROC principal
        fig_roc_alt.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", 
            line=dict(color="blue", width=2), 
            name=f"AUC = {roc_auc:.2f}"
        ))

        # Línea de azar
        fig_roc_alt.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", 
            line=dict(dash="dash", color="red", width=1), 
            name="Línea de Azar"
        ))

        # Personalización del diseño
        fig_roc_alt.update_layout(
            title=dict(
                text="Curva ROC con Área Sombreada",
                font=dict(size=16)
            ),
            xaxis=dict(
                title="Tasa de Falsos Positivos (FPR)", 
                tickfont=dict(size=12),
                title_font=dict(size=14)
            ),
            yaxis=dict(
                title="Tasa de Verdaderos Positivos (TPR)", 
                tickfont=dict(size=12),
                title_font=dict(size=14)
            ),
            legend=dict(
                title="Leyenda",
                font=dict(size=12),
                x=0.7, y=0.2
            ),
            plot_bgcolor="rgba(240, 240, 240, 0.8)",
            width=800, height=500
        )

        # Mostrar el gráfico
        st.plotly_chart(fig_roc_alt, use_container_width=True)

        # Análisis de Residuos
        st.markdown("### 📉 Análisis de Residuos")
        residuals = st.session_state.y_test - y_pred_proba
        fig_resid = px.histogram(
            residuals, nbins=50, 
            title="Distribución de Residuos",
            color_discrete_sequence=["#007acc"]
        )
        fig_resid.update_layout(
            xaxis_title="Residuos",
            yaxis_title="Frecuencia",
            margin=dict(t=50, b=30)
        )
        st.plotly_chart(fig_resid, use_container_width=True)

        # Frontera de Decisión
        if len(st.session_state.feature_variables) == 2:
            st.markdown("### 🌐 Visualización de la Frontera de Decisión")

            # Crear una malla para representar las regiones de decisión
            x_min, x_max = st.session_state.X_train_scaled[:, 0].min() - 1, st.session_state.X_train_scaled[:, 0].max() + 1
            y_min, y_max = st.session_state.X_train_scaled[:, 1].min() - 1, st.session_state.X_train_scaled[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                 np.arange(y_min, y_max, 0.02))

            Z = st.session_state.model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            fig_decision = px.imshow(
                Z, x=xx[0], y=yy[:, 0], origin='lower',
                aspect='auto', color_continuous_scale='Spectral',
                title='Frontera de Decisión'
            )
            fig_decision.update_layout(
                xaxis_title=st.session_state.feature_variables[0],
                yaxis_title=st.session_state.feature_variables[1]
            )
            fig_decision.update_traces(opacity=0.5)

            # Añadir puntos de entrenamiento
            fig_decision.add_scatter(
                x=st.session_state.X_train_scaled[:, 0],
                y=st.session_state.X_train_scaled[:, 1],
                mode='markers',
                marker=dict(
                    color=st.session_state.y_train,
                    colorscale='Spectral',
                    line=dict(color='black', width=1)
                ),
                name='Datos de Entrenamiento'
            )
            st.plotly_chart(fig_decision, use_container_width=True)
        else:
            st.info("La visualización de la frontera de decisión solo está disponible para dos variables independientes.")

        # Validación Cruzada
        st.markdown("### 🔄 Validación Cruzada")
        cv_scores = cross_val_score(
            st.session_state.model, 
            st.session_state.scaler.transform(st.session_state.X), 
            st.session_state.y, cv=5, scoring='accuracy'
        )
        st.write(f"**Puntuaciones de validación cruzada:** {cv_scores}")
        st.success(f"**Precisión media:** {cv_scores.mean():.2f}")


    # Paso 6: Predicción con Nuevos Datos
elif step == "🔮 Paso 6: Predicción con Nuevos Datos":
    st.header("🔮 Paso 6: Predicción con Nuevos Datos")

    if "model" not in st.session_state or st.session_state.model is None:
        st.warning("⚠️ Por favor, entrena el modelo en el Paso 4 antes de continuar.")
    else:
        new_data = {}
        st.markdown("### 🖊️ Ingrese los valores para las variables independientes")
        for feature in st.session_state.feature_variables:
            value = st.number_input(f"**Valor para {feature}:**", value=0.0, step=0.1)
            new_data[feature] = [value]

        if st.button("✨ Realizar Predicción"):
            # Crear DataFrame con los datos ingresados
            new_df = pd.DataFrame(new_data)
            new_df_scaled = st.session_state.scaler.transform(new_df)

            # Realizar predicción
            prediction = st.session_state.model.predict(new_df_scaled)
            prediction_proba = st.session_state.model.predict_proba(new_df_scaled)

            # Mostrar resultados
            st.markdown("### 🧾 Resultados de la Predicción")
            st.success(f"**Predicción:** {'Clase 1 (Positiva)' if prediction[0] == 1 else 'Clase 0 (Negativa)'}")
            st.info(f"**Probabilidad de Clase 1:** {prediction_proba[0][1]:.2f}")
            st.info(f"**Probabilidad de Clase 0:** {prediction_proba[0][0]:.2f}")

            # Visualización de la función sigmoide
            st.markdown("### 📈 Visualización de la Función Sigmoide")
            z = np.linspace(-10, 10, 200)
            sigmoid = 1 / (1 + np.exp(-z))
            z_new_example = np.dot(new_df_scaled, st.session_state.model.coef_.T) + st.session_state.model.intercept_

            fig_sigmoid = go.Figure()

            # Línea de la función sigmoide
            fig_sigmoid.add_trace(go.Scatter(
                x=z, y=sigmoid, mode='lines',
                name='Función Sigmoide',
                line=dict(color="blue", width=2)
            ))

            # Punto del nuevo ejemplo
            fig_sigmoid.add_trace(go.Scatter(
                x=z_new_example.flatten(),
                y=1 / (1 + np.exp(-z_new_example.flatten())),
                mode='markers', name='Nuevo Ejemplo',
                marker=dict(color='red', size=10)
            ))

            fig_sigmoid.update_layout(
                title="Función Sigmoide y Nuevo Ejemplo",
                xaxis_title='Valor de Z',
                yaxis_title='Probabilidad',
                showlegend=True,
                plot_bgcolor="rgba(240,240,240,0.9)"
            )
            st.plotly_chart(fig_sigmoid, use_container_width=True)

