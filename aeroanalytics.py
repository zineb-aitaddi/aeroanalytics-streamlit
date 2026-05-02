# =============================================================
#  Aeronautics Dashboard - Version Luxe ++ PDF ReportLab
#  Auteur : MED
#  Dataset par défaut : flights_aero.csv
# ============================================================

from io import BytesIO
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
import plotly.graph_objects as go # <--- NOUVEL IMPORT NÉCESSAIRE POUR LES JAUGE et GRAPHIQUES DE COMPARAISON
import wikipedia # <--- NOUVEL IMPORT NÉCESSAIRE POUR LA DESCRIPTION DÉTAILLÉE

# ------------------------------------------------------------
# CONFIGURATION GLOBALE
# ------------------------------------------------------------
st.set_page_config(
    page_title="Aeronautics Analytics",
    page_icon="✈️",
    layout="wide",
)

DATA_PATH = Path("flights_aero.csv")
LOGO_PATH = Path("logo.png")

# chemins pour le modèle de reconnaissance d’avions
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "aircraft_resnet18.pth"
CLASS_INDEX_PATH = MODELS_DIR / "class_to_idx.json"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ------------------------------------------------------------
# FONCTIONS UTILITAIRES
# ------------------------------------------------------------
def get_numeric_cols(df: pd.DataFrame) -> list[str]:
    base = [
        "distance_km",
        "duration_min",
        "delay_min",
        "cruise_altitude_ft",
        "cruise_speed_kts",
        "fuel_burn_kg",
        "co2_kg",
        "load_factor",
        "passengers",
    ]
    return [c for c in base if c in df.columns]


@st.cache_data
def load_csv_from_path(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data
def load_csv_from_upload(upload) -> pd.DataFrame:
    df = pd.read_csv(upload)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


# Noms professionnels
DISPLAY_NAMES = {
    "distance_km": "Distance (km)",
    "duration_min": "Durée du vol (min)",
    "delay_min": "Retard (min)",
    "cruise_altitude_ft": "Altitude de croisière (ft)",
    "cruise_speed_kts": "Vitesse de croisière (kts)",
    "fuel_burn_kg": "Consommation carburant (kg)",
    "co2_kg": "Émissions CO₂ (kg)",
    "load_factor": "Taux d’occupation (%)",
    "passengers": "Passagers",
}
CATEGORICAL_DISPLAY = {
    "airline": "Compagnie",
    "aircraft_type": "Type d’avion",
    "origin": "Origine",
    "destination": "Destination",
    "route": "Route",
}


def pretty(name: str) -> str:
    return DISPLAY_NAMES.get(name, CATEGORICAL_DISPLAY.get(name, name))


# ------------------------------------------------------------
#  MODÈLE DE RECONNAISSANCE D’AVION (ResNet18)
# ------------------------------------------------------------
@st.cache_resource(show_spinner="Chargement du modèle de reconnaissance d’avions…")
def load_aircraft_model():
    """
    Charge le modèle ResNet18 entraîné sur FGVC-Aircraft (CPU)
    et retourne (model, idx_to_class).
    """
    if not MODEL_PATH.exists():
        st.error(f"⚠️ Fichier modèle introuvable : {MODEL_PATH}")
        return None, None

    if not CLASS_INDEX_PATH.exists():
        st.error(f"⚠️ Fichier de classes introuvable : {CLASS_INDEX_PATH}")
        return None, None

    # class_to_idx : {"Boeing 737-800": 0, ...}
    with open(CLASS_INDEX_PATH, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    num_classes = len(idx_to_class)

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()  # mode évaluation

    return model, idx_to_class


def preprocess_aircraft_image(file):
    """
    Prépare une image uploadée pour le modèle.
    Retourne (tensor, image_PIL).
    """
    img = Image.open(file).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    tensor = transform(img).unsqueeze(0)  # (1, 3, 224, 224)
    return tensor, img


def predict_aircraft(file, top_k: int = 3):
    """
    Retourne (liste de (label, prob), image_PIL).
    """
    model, idx_to_class = load_aircraft_model()
    if model is None:
        return [], None

    x, pil_img = preprocess_aircraft_image(file)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    top_probs, top_idxs = torch.topk(probs, k=top_k)

    results = []
    for p, idx in zip(top_probs, top_idxs):
        label = idx_to_class[int(idx)]
        results.append((label, float(p)))

    return results, pil_img
 #------------------------------------------------------------
#  MODÈLE DE RECONNAISSANCE D’AVION (ResNet18)
# ------------------------------------------------------------
@st.cache_resource(show_spinner="Chargement du modèle de reconnaissance d’avions…")
def load_aircraft_model():
    """
    Charge le modèle ResNet18 entraîné sur FGVC-Aircraft (CPU)
    et retourne (model, idx_to_class).
    """
    if not MODEL_PATH.exists():
        st.error(f"⚠️ Fichier modèle introuvable : {MODEL_PATH}")
        return None, None

    if not CLASS_INDEX_PATH.exists():
        st.error(f"⚠️ Fichier de classes introuvable : {CLASS_INDEX_PATH}")
        return None, None

    # class_to_idx : {"Boeing 737-800": 0, ...}
    with open(CLASS_INDEX_PATH, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    num_classes = len(idx_to_class)

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()  # mode évaluation

    return model, idx_to_class


def preprocess_aircraft_image(file):
    """
    Prépare une image uploadée pour le modèle.
    Retourne (tensor, image_PIL).
    """
    img = Image.open(file).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    tensor = transform(img).unsqueeze(0)  # (1, 3, 224, 224)
    return tensor, img


def predict_aircraft(file, top_k: int = 3):
    """
    Retourne (liste de (label, prob), image_PIL).
    """
    model, idx_to_class = load_aircraft_model()
    if model is None:
        return [], None

    x, pil_img = preprocess_aircraft_image(file)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    top_probs, top_idxs = torch.topk(probs, k=top_k)

    results = []
    for p, idx in zip(top_probs, top_idxs):
        label = idx_to_class[int(idx)]
        results.append((label, float(p)))

    return results, pil_img

# ------------------------------------------------------------
# DONNÉES TECHNIQUES SIMULÉES ET FONCTION DE RÉCUPÉRATION
# ------------------------------------------------------------
# ATTENTION: En production, ces données devraient provenir d'une base de données 
# structurée. Ceci est une SIMULATION manuelle basée sur des données publiques.

SIMULATED_SPECS = {
    # GROS PORTEURS
    "Airbus A380": {
        "Vitesse_max_kmh": 1020, 
        "Poids_max_decollage_kg": 575000, 
        "Vitesse_croisiere_kmh": 945,
        "Concurrents": {"Boeing 747-400": 920, "Boeing 777-300ER": 905}
    },
    "Boeing 747-400": {
        "Vitesse_max_kmh": 988, 
        "Poids_max_decollage_kg": 396890, 
        "Vitesse_croisiere_kmh": 920,
        "Concurrents": {"Airbus A380": 945, "Boeing 777-300ER": 905}
    },
    "Boeing 777-300ER": {
        "Vitesse_max_kmh": 950, 
        "Poids_max_decollage_kg": 351500, 
        "Vitesse_croisiere_kmh": 905,
        "Concurrents": {"Airbus A350-900": 903, "Boeing 787-9": 903}
    },
    "Airbus A330-300": {
        "Vitesse_max_kmh": 918, 
        "Poids_max_decollage_kg": 233000, 
        "Vitesse_croisiere_kmh": 871,
        "Concurrents": {"Boeing 787-8": 890, "Boeing 767-400": 850}
    },
    "Boeing 787-9": {
        "Vitesse_max_kmh": 954, 
        "Poids_max_decollage_kg": 254000, 
        "Vitesse_croisiere_kmh": 903,
        "Concurrents": {"Airbus A350-900": 903, "Airbus A330-300": 871}
    },
    "Airbus A350-900": {
        "Vitesse_max_kmh": 945, 
        "Poids_max_decollage_kg": 280000, 
        "Vitesse_croisiere_kmh": 903,
        "Concurrents": {"Boeing 787-9": 903, "Boeing 777-300ER": 905}
    },

    # MONOCOULOIRS
    "Boeing 737-800": {
        "Vitesse_max_kmh": 946, 
        "Poids_max_decollage_kg": 79000, 
        "Vitesse_croisiere_kmh": 828,
        "Concurrents": {"Airbus A320": 828, "Boeing 737-700": 790}
    },
    "Airbus A320": {
        "Vitesse_max_kmh": 903, 
        "Poids_max_decollage_kg": 78000, 
        "Vitesse_croisiere_kmh": 828,
        "Concurrents": {"Boeing 737-800": 828, "Airbus A321": 850}
    },
    "Airbus A321": {
        "Vitesse_max_kmh": 903, 
        "Poids_max_decollage_kg": 93500, 
        "Vitesse_croisiere_kmh": 850,
        "Concurrents": {"Boeing 737-900": 840, "Airbus A320": 828}
    },

    # AVIONS RÉGIONAUX
    "Embraer E190": {
        "Vitesse_max_kmh": 870, 
        "Poids_max_decollage_kg": 51800, 
        "Vitesse_croisiere_kmh": 830,
        "Concurrents": {"Bombardier CRJ900": 829, "ATR 72": 510}
    },
    # REMPLACEZ/AJOUTEZ LES NOMS CI-DESSUS pour qu'ils correspondent 
    # aux classes de votre modèle ResNet18
}

def get_aircraft_specs(aircraft_name):
    """
    Simule la récupération des spécifications techniques à partir de SIMULATED_SPECS.
    """
    # Recherche du nom exact ou d'une correspondance partielle 
    for key, specs in SIMULATED_SPECS.items():
        if aircraft_name in key or key in aircraft_name:
            return specs
            
    # Cas par défaut si le modèle n'est pas trouvé dans la base simulée
    return {
        "Vitesse_max_kmh": "N/A",
        "Poids_max_decollage_kg": "N/A",
        "Vitesse_croisiere_kmh": "N/A",
        "Concurrents": {}
    }

# ------------------------------------------------------------
# NOUVELLE FONCTION POUR WIKIPEDIA
# ------------------------------------------------------------
@st.cache_data(show_spinner="Recherche d'informations Wikipedia...")
def get_wiki_data(aircraft_name):
    """
    Récupère des informations de Wikipedia pour un avion donné.
    Retourne (summary: str, url: str) ou (None, None).
    """
    try:
        # Configuration pour la langue française
        import wikipedia # Nécessaire pour le cache
        wikipedia.set_lang("fr")

        # Tentative de recherche
        search_results = wikipedia.search(aircraft_name, results=1)
        if not search_results:
            # Si pas de résultat en FR, tenter en EN
            wikipedia.set_lang("en")
            search_results = wikipedia.search(aircraft_name, results=1)
            if not search_results:
                return None, None

        page_title = search_results[0]
        # Utiliser auto_suggest=True pour corriger les fautes de frappe
        page = wikipedia.page(page_title, auto_suggest=True)

        # Résumé pour la description
        summary = page.summary

        # URL pour le lien direct
        url = page.url

        return summary, url

    except wikipedia.exceptions.PageError:
        # La page n'existe pas ou le titre n'est pas correct
        return None, None
    except wikipedia.exceptions.DisambiguationError:
        # Ambiguïté (e.g., "Airbus" peut renvoyer plusieurs pages)
        return None, None
    except Exception as e:
        # Gérer les autres erreurs (connexion, etc.)
        return None, None



def render_aircraft_recognition_tab():
    """
    Onglet Streamlit : reconnaissance d’avion par image, avec fiche technique Wikipedia.
    """
    st.markdown("### 🔍 Reconnaissance d’avion par image")
    st.markdown(
        """
        Importez une photo d’avion (idéalement vue latérale / 3-quart) et le modèle
        tentera de reconnaître la **famille d’appareil** à partir du dataset FGVC-Aircraft.
        """
    )

    uploaded_file = st.file_uploader(
        "Choisissez une image (JPG ou PNG)", type=["jpg", "jpeg", "png"]
    )

    if not uploaded_file:
        st.info("📥 Uploade une image pour lancer la détection.")
        return

    # 1. Prédiction du modèle
    with st.spinner("Analyse de l’image en cours…"):
        preds, pil_img = predict_aircraft(uploaded_file)

    if pil_img is None or not preds:
        st.error("Impossible de générer une prédiction. Vérifiez le modèle et les fichiers.")
        return

    best_label, best_prob = preds[0]
    
    # 2. Affichage des résultats et de l'image
    col_img, col_res = st.columns([1.2, 1])

    with col_img:
        st.markdown("#### Image analysée")
        st.image(pil_img, use_container_width=True)

    with col_res:
        st.markdown("#### Identification")
        st.success(
            f"**Avion prédit :** {best_label}  \n"
            f"**Confiance :** {best_prob * 100:.1f} %"
        )
        
        # Graphe pour la confiance (utilisant plotly.graph_objects comme dans la structure initiale)
        labels = [p[0] for p in preds]
        probs = [p[1] for p in preds]
        
        fig_conf = go.Figure(
            data=[
                go.Bar(
                    y=[f"{p*100:.1f}%"], 
                    x=[l], 
                    marker_color=['#10b981' if i == 0 else '#6b7280' for i in range(len(labels))]
                ) for i, (l, p) in enumerate(zip(labels, probs))
            ],
            layout=go.Layout(
                title='Top 3 Prédictions',
                xaxis={'title': 'Type d\'avion'},
                yaxis={'visible': False},
                height=200,
                margin=dict(l=20, r=20, t=40, b=20),
            )
        )
        st.plotly_chart(fig_conf, use_container_width=True)

    # 3. Fiche Technique Généraliste (Wikipedia)
    st.markdown("---")
    st.header("📄 Fiche Technique Détaillée")
    
    # Séparer Constructeur / Modèle du label prédit
    parts = best_label.split()
    manufacturer = parts[0]
    model_name = " ".join(parts[1:]) if len(parts) > 1 else "Modèle non précisé"

    col_manuf, col_model = st.columns(2)
    with col_manuf:
        st.metric("**Constructeur estimé**", manufacturer)
    with col_model:
        st.metric("**Famille / Variante**", model_name)
    
    st.markdown("---")
    
    # Appel de la fonction Wikipedia
    summary, wiki_url = get_wiki_data(best_label)
    
    if summary:
        st.subheader(f"Description : {best_label}")
        # Afficher le résumé comme une citation pour le mettre en évidence
        st.markdown(f"> **{summary}**")
        
        st.markdown(
            f"**Source des données :** [Article Wikipedia de l'avion]({wiki_url})"
        )
        st.caption("✅ Les informations sont récupérées en temps réel depuis Wikipedia, assurant une bonne fraîcheur.")
        
    else:
        st.warning(
            f"❌ Aucune fiche technique détaillée n'a pu être trouvée sur Wikipedia "
            f"pour **{best_label}** ou le format du nom n'est pas standard."
        )

    # 4. Spécifications Techniques (Vitesse & Poids)
    st.markdown("---")
    
    specs = get_aircraft_specs(best_label) # Utilise la fonction de simulation
    
    st.subheader("🚀 Spécifications Techniques Clés (Simulées)")

    col_speed, col_weight, col_alt = st.columns(3)

    with col_speed:
        st.metric(
            "**Vitesse de Croisière**",
            f"{specs['Vitesse_croisiere_kmh']} km/h" if specs['Vitesse_croisiere_kmh'] != 'N/A' else "N/A",
            help="Vitesse typique en vol de croisière."
        )
    with col_weight:
        # Ajout du formatage pour les grands nombres avec séparateur de milliers
        poids_format = f"{specs['Poids_max_decollage_kg']:,.0f} kg" if isinstance(specs['Poids_max_decollage_kg'], int) else "N/A"
        st.metric(
            "**Poids Max. Décollage**",
            poids_format.replace(',', ' '), # Remplacement du séparateur pour la lisibilité
            help="Masse maximale au décollage (MTOW)."
        )
    with col_alt:
        st.metric(
            "**Vitesse Max. Opérationnelle**",
            f"{specs['Vitesse_max_kmh']} km/h" if specs['Vitesse_max_kmh'] != 'N/A' else "N/A",
            help="Vitesse maximale à ne pas dépasser (MMO)."
        )
        
    st.markdown("---")

    # 5. Graphe selon l'avion détecté (Comparaison de vitesse)
    competitors = specs.get("Concurrents", {})
    if competitors and specs['Vitesse_croisiere_kmh'] != 'N/A' and isinstance(specs['Vitesse_croisiere_kmh'], int):
        
        st.subheader(f"📊 Comparaison de Vitesse de Croisière")
        
        # Préparation des données pour le graphe
        data = [{"Avion": best_label, "Vitesse (km/h)": specs["Vitesse_croisiere_kmh"], "Catégorie": "Sélectionné"}]
        for competitor, speed in competitors.items():
            data.append({"Avion": competitor, "Vitesse (km/h)": speed, "Catégorie": "Concurrent"})

        # Conversion en DataFrame
        df_comp = pd.DataFrame(data)
        
        # Création du graphique à barres (style Streamlit/Plotly préservé)
        fig_comp_speed = px.bar(
            df_comp.sort_values("Vitesse (km/h)", ascending=True),
            x="Vitesse (km/h)",
            y="Avion",
            orientation="h",
            color="Catégorie",
            color_discrete_map={'Sélectionné': '#10b981', 'Concurrent': '#6b7280'},
            labels={"Avion": "", "Vitesse (km/h)": "Vitesse de Croisière (km/h)"},
            title=f"Vitesse de Croisière : {best_label} vs. Concurrents",
        )
        fig_comp_speed.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
        st.plotly_chart(fig_comp_speed, use_container_width=True)
        
    elif specs['Vitesse_croisiere_kmh'] == 'N/A':
        st.info(
            f"❌ Aucune spécification technique simulée trouvée pour **{best_label}**. "
            f"Veuillez ajouter ce modèle à la base de données simulée `SIMULATED_SPECS` pour afficher le graphe et les specs."
        )
    else:
        st.info(f"Pas de données de comparaison définies dans `SIMULATED_SPECS` pour {best_label}.")
        


# ------------------------------------------------------------
# CHARGEMENT DATASET PAR DÉFAUT
# ------------------------------------------------------------
if not DATA_PATH.exists():
    st.error(f"⚠️ Fichier de données introuvable : {DATA_PATH.resolve()}")
    st.stop()

df = load_csv_from_path(DATA_PATH)
NUM_COLS = get_numeric_cols(df)

# ------------------------------------------------------------
# SIDEBAR : LOGO, THEME, FILTRES, UPLOAD, MODE VUE
# ------------------------------------------------------------
with st.sidebar:
    # Logo
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=180)
    else:
        st.markdown("### ✈️ AirAnalytics")

    # Thème
    theme_mode = st.radio(
        "Thème",
        ["Sombre", "Clair"],
        index=0,
        horizontal=True,
    )

    # Carte filtres
    st.markdown(
        """
        <div class="filter-card">
            <div class="filter-title">Filtres principaux</div>
        """,
        unsafe_allow_html=True,
    )

    airline_options = ["Toutes"] + sorted(df["airline"].dropna().unique().tolist())
    selected_airline = st.selectbox("✈ Compagnie aérienne", airline_options)

    aircraft_options = ["Tous"] + sorted(df["aircraft_type"].dropna().unique().tolist())
    selected_aircraft = st.selectbox("🛩 Type d’avion", aircraft_options)

    if "date" in df.columns:
        min_date = df["date"].min()
        max_date = df["date"].max()
        date_range = st.date_input(
            "📅 Période",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
    else:
        date_range = None

    if "distance_km" in df.columns:
        min_dist, max_dist = float(df["distance_km"].min()), float(df["distance_km"].max())
        dist_min, dist_max = st.slider(
            "📏 Distance (km)",
            min_value=min_dist,
            max_value=max_dist,
            value=(min_dist, max_dist),
        )
    else:
        dist_min = dist_max = None

    if "duration_min" in df.columns:
        min_dur, max_dur = float(df["duration_min"].min()), float(df["duration_min"].max())
        dur_min, dur_max = st.slider(
            "⏱ Durée (min)",
            min_value=min_dur,
            max_value=max_dur,
            value=(min_dur, max_dur),
        )
    else:
        dur_min = dur_max = None

    st.markdown(
        """
        <hr style="margin-top: 1.1rem; margin-bottom: 0.6rem; border:none; border-top:1px dashed rgba(148,163,184,0.7);" />
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Upload CSV (source alternative)
    st.markdown("##### 📁 Importer un nouveau fichier")
    uploaded_file = st.file_uploader("CSV de vols / avions", type=["csv"])
    use_uploaded = False
    if uploaded_file is not None:
        try:
            df = load_csv_from_upload(uploaded_file)
            NUM_COLS = get_numeric_cols(df)
            use_uploaded = True
            st.success("Nouveau fichier chargé avec succès ✅")
        except Exception as e:
            st.error(f"Erreur de lecture du fichier : {e}")

    show_raw = st.checkbox("Afficher les données filtrées", value=False)

    # Mode d'affichage
    view_mode = st.radio(
        "Mode d'affichage",
        ["Vue compacte", "Vue détaillée"],
        index=0,
        horizontal=False,
    )

# ------------------------------------------------------------
# THEME CSS
# ------------------------------------------------------------
if theme_mode == "Sombre":
    main_bg = "#020617"
    main_text = "#e5e7eb"
    card_bg = "#020617"
    sidebar_bg = "#020617"
    sidebar_text = "#f9fafb"
    button_bg = "#111827"
else:
    main_bg = "#f5f5f7"
    main_text = "#020617"
    card_bg = "#ffffff"
    sidebar_bg = "#f9fafb"
    sidebar_text = "#111827"
    button_bg = "#ffffff"

st.markdown(
    f"""
    <style>
    :root {{
        --font-main: "Inter", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    html, body, [class*="css"] {{
        font-family: var(--font-main) !important;
    }}
    .main {{
        background-color: {main_bg};
        color: {main_text};
    }}
    .block-container {{
        padding-top: 1.4rem;
        padding-bottom: 1.4rem;
        max-width: 1400px;
    }}
    h1, h2, h3, h4 {{
        color: {main_text} !important;
    }}

    [data-testid="stSidebar"] {{
        background: {sidebar_bg};
        color: {sidebar_text};
        border-right: 1px solid rgba(148,163,184,0.55);
    }}
    [data-testid="stSidebar"] * {{
        font-family: var(--font-main) !important;
        color: {sidebar_text};
    }}

    .filter-card {{
        background: #ffffff;
        border-radius: 1rem;
        padding: 0.9rem 0.9rem 0.6rem 0.9rem;
        border: 1px solid rgba(148,163,184,0.6);
        margin-bottom: 0.8rem;
        box-shadow: 0 10px 25px rgba(15,23,42,0.15);
    }}
    .filter-title {{
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        margin-bottom: 0.6rem;
        color: {sidebar_text};
    }}

    [data-testid="stSidebar"] label {{
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }}

    [data-baseweb="select"] > div {{
        border-radius: 0.85rem !important;
        border: 1px solid rgba(148,163,184,0.85) !important;
        box-shadow: 0 6px 16px rgba(15,23,42,0.18) !important;
        background-color: #f9fafb !important;
    }}

    .metric-card {{
        background: {card_bg};
        border-radius: 1rem;
        padding: 1rem 1.2rem;
        border: 1px solid rgba(148,163,184,0.4);
        box-shadow: 0 18px 40px rgba(15,23,42,0.35);
    }}
    .metric-label {{
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: rgba(148,163,184,0.9);
    }}
    .metric-value {{
        font-size: 1.5rem;
        font-weight: 600;
    }}
    .metric-sub {{
        font-size: 0.75rem;
        color: rgba(148,163,184,0.9);
    }}

    .axis-card {{
        background: {card_bg};
        border-radius: 1rem;
        padding: 0.9rem 1rem 0.4rem 1rem;
        border: 1px solid rgba(148,163,184,0.45);
        box-shadow: 0 16px 35px rgba(15,23,42,0.35);
        margin-bottom: 0.7rem;
    }}
    .axis-title {{
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: rgba(148,163,184,0.9);
        margin-bottom: 0.4rem;
    }}

    .filter-chip {{
        display: inline-flex;
        align-items: center;
        padding: 0.15rem 0.7rem;
        margin-right: 0.35rem;
        margin-bottom: 0.25rem;
        border-radius: 999px;
        border: 1px solid rgba(148,163,184,0.7);
        font-size: 0.75rem;
        background: rgba(15,23,42,0.03);
    }}

    .export-card {{
        background: {card_bg};
        border-radius: 1rem;
        padding: 0.9rem 1rem 0.8rem 1rem;
        border: 1px solid rgba(148,163,184,0.5);
        box-shadow: 0 12px 30px rgba(15,23,42,0.25);
        margin-top: 0.7rem;
        margin-bottom: 0.7rem;
    }}
    .export-title {{
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: rgba(148,163,184,0.9);
        margin-bottom: 0.4rem;
    }}

    button[kind="secondary"],
    button[kind="primary"],
    button[data-testid="baseButton-secondary"],
    button[data-testid="baseButton-primary"] {{
        border-radius: 999px !important;
        padding: 0.5rem 1.3rem !important;
        border: 1px solid rgba(148,163,184,0.8) !important;
        background-color: {button_bg} !important;
        color: {main_text} !important;
        font-weight: 500 !important;
        box-shadow: 0 8px 18px rgba(15,23,42,0.15) !important;
    }}
    button:hover {{
        border-color: rgba(59,130,246,0.8) !important;
        box-shadow: 0 10px 22px rgba(37,99,235,0.25) !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# APPLICATION DES FILTRES
# ------------------------------------------------------------
mask = pd.Series(True, index=df.index)

if selected_airline != "Toutes":
    mask &= df["airline"] == selected_airline
if selected_aircraft != "Tous":
    mask &= df["aircraft_type"] == selected_aircraft

if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    if "date" in df.columns:
        mask &= df["date"].between(start, end)

if dist_min is not None and dist_max is not None and "distance_km" in df.columns:
    mask &= df["distance_km"].between(dist_min, dist_max)

if dur_min is not None and dur_max is not None and "duration_min" in df.columns:
    mask &= df["duration_min"].between(dur_min, dur_max)

fdf = df.loc[mask].copy()
if fdf.empty:
    st.warning("Aucune donnée ne correspond aux filtres sélectionnés.")
    st.stop()

NUM_COLS = get_numeric_cols(fdf)

# ------------------------------------------------------------
# KPIs globaux
# ------------------------------------------------------------
kpi_total_vols = len(fdf)
kpi_nb_comp = fdf["airline"].nunique() if "airline" in fdf.columns else None
kpi_nb_types = fdf["aircraft_type"].nunique() if "aircraft_type" in fdf.columns else None
kpi_delay_mean = fdf["delay_min"].mean() if "delay_min" in fdf.columns else None
kpi_delay_med = fdf["delay_min"].median() if "delay_min" in fdf.columns else None
kpi_fuel_mean = fdf["fuel_burn_kg"].mean() if "fuel_burn_kg" in fdf.columns else None
kpi_fuel_sum = fdf["fuel_burn_kg"].sum() if "fuel_burn_kg" in fdf.columns else None
kpi_co2_mean = fdf["co2_kg"].mean() if "co2_kg" in fdf.columns else None
kpi_co2_sum = fdf["co2_kg"].sum() if "co2_kg" in fdf.columns else None
kpi_dist_mean = fdf["distance_km"].mean() if "distance_km" in fdf.columns else None
kpi_dur_mean = fdf["duration_min"].mean() if "duration_min" in fdf.columns else None

# ------------------------------------------------------------
# PDF REPORTLAB : FIGS + TEXTE CONSULTANT
# ------------------------------------------------------------
def fig_to_png_bytes(fig, width=900, height=400, scale=2) -> bytes:
    return fig.to_image(format="png", width=width, height=height, scale=scale)


def generate_pdf_report(df_filtered: pd.DataFrame, filters_text: list[str]) -> bytes:
    """
    Rapport PDF détaillé avec :
    - Executive summary
    - Graphique retard dans le temps + explication
    - Graphique retard par compagnie + explication
    - Graphique top routes + explication
    - Analyse environnement
    - Recommandations
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm
    from reportlab.lib.utils import ImageReader
    from textwrap import wrap

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    def write_line(txt, x, y, size=10, bold=False):
        font = "Helvetica-Bold" if bold else "Helvetica"
        c.setFont(font, size)
        c.drawString(x, y, txt)

    def write_paragraph(txt, x, y, size=10, bold=False, max_width=16 * cm):
        font = "Helvetica-Bold" if bold else "Helvetica"
        c.setFont(font, size)
        chars_per_line = int(max_width / (size * 0.5))
        lines = wrap(txt, width=chars_per_line)
        for line in lines:
            c.drawString(x, y, line)
            y -= 0.4 * cm
        return y

    # Figures
    ts_png = None
    airline_png = None
    routes_png = None

    # 1) Évolution retard
    if {"date", "delay_min"}.issubset(df_filtered.columns):
        ts = (
            df_filtered.groupby("date", as_index=False)["delay_min"]
            .mean()
            .sort_values("date")
        )
        if not ts.empty:
            fig_ts = px.line(
                ts,
                x="date",
                y="delay_min",
                labels={"date": "Date", "delay_min": "Retard moyen (min)"},
                title="Évolution du retard moyen",
            )
            ts_png = fig_to_png_bytes(fig_ts)

    # 2) Retard moyen par compagnie
    if {"airline", "delay_min"}.issubset(df_filtered.columns):
        agg = (
            df_filtered.groupby("airline", as_index=False)["delay_min"]
            .mean()
            .sort_values("delay_min", ascending=False)
        )
        if not agg.empty:
            fig_air = px.bar(
                agg,
                x="airline",
                y="delay_min",
                labels={"airline": "Compagnie", "delay_min": "Retard moyen (min)"},
                title="Retard moyen par compagnie",
            )
            airline_png = fig_to_png_bytes(fig_air)

    # 3) Top routes
    if {"origin", "destination"}.issubset(df_filtered.columns):
        df_tmp = df_filtered.copy()
        df_tmp["route"] = df_tmp["origin"] + " → " + df_tmp["destination"]
        simple_counts = (
            df_tmp["route"]
            .value_counts()
            .head(12)
            .reset_index(name="vols_count")
            .rename(columns={"index": "route"})
        )
        if not simple_counts.empty:
            fig_routes = px.bar(
                simple_counts,
                x="vols_count",
                y="route",
                orientation="h",
                labels={"vols_count": "Nombre de vols", "route": "Route"},
                title="Top routes par nombre de vols",
            )
            routes_png = fig_to_png_bytes(fig_routes)

    # Analyses textuelles
    source_txt = "fichier importé" if use_uploaded else "fichier flights_aero.csv"

    tendance_txt = ""
    if {"date", "delay_min"}.issubset(df_filtered.columns):
        ts2 = (
            df_filtered.groupby("date", as_index=False)["delay_min"]
            .mean()
            .sort_values("date")
        )
        if len(ts2) >= 4:
            n = len(ts2)
            early = ts2["delay_min"].iloc[: max(1, n // 4)].mean()
            late = ts2["delay_min"].iloc[-max(1, n // 4):].mean()
            diff = late - early
            if diff > 0.2:
                tendance_txt = (
                    f"On observe une dégradation progressive du retard moyen sur la période (+{diff:.1f} min), "
                    "ce qui suggère une tension croissante sur l'exploitation (saisonnalité, saturation, ressources)."
                )
            elif diff < -0.2:
                tendance_txt = (
                    f"On observe une amélioration du retard moyen sur la période ({diff:.1f} min), "
                    "ce qui traduit un effet positif des actions correctives ou une période moins contrainte."
                )
            else:
                tendance_txt = (
                    "Le retard moyen reste globalement stable sur la période, "
                    "sans évolution significative."
                )

    seg_txt = ""
    if {"distance_km", "delay_min"}.issubset(df_filtered.columns):
        short = df_filtered[df_filtered["distance_km"] <= 1000]["delay_min"].mean()
        medium = df_filtered[
            (df_filtered["distance_km"] > 1000) & (df_filtered["distance_km"] <= 3000)
        ]["delay_min"].mean()
        long = df_filtered[df_filtered["distance_km"] > 3000]["delay_min"].mean()
        seg_txt = (
            f"Les vols courts (≤ 1 000 km) présentent un retard moyen de {short:.1f} min, "
            f"contre {medium:.1f} min pour les vols moyens (1 000–3 000 km) et {long:.1f} min "
            "pour les vols long-courriers (> 3 000 km). "
            "Cela met en évidence un impact plus marqué des contraintes opérationnelles sur le long-courrier."
        )

    env_txt = ""
    if {"co2_kg", "fuel_burn_kg"}.issubset(df_filtered.columns):
        total_co2 = df_filtered["co2_kg"].sum()
        mean_co2 = df_filtered["co2_kg"].mean()
        total_fuel = df_filtered["fuel_burn_kg"].sum()
        env_txt = (
            f"L'empreinte carbone totale associée aux vols de la période est estimée à {total_co2:,.0f} kg de CO₂, "
            f"soit en moyenne {mean_co2:.0f} kg de CO₂ par vol. "
            f"La consommation totale de carburant est de {total_fuel:,.0f} kg. "
            "La réduction des retards et l'optimisation du profil de vol constituent des leviers directs "
            "pour diminuer cet impact."
        )

    # -------- Page 1 : Executive Summary --------
    from reportlab.lib.pagesizes import A4 as _A4
    from reportlab.lib.units import cm

    y = height - 2 * cm
    write_line("AirAnalytics – Rapport d'analyse des vols", 2 * cm, y, size=18, bold=True)
    y -= 1.2 * cm

    y = write_paragraph(
        "Ce rapport présente une analyse détaillée des performances opérationnelles, du réseau de routes "
        "et de l'impact environnemental des vols observés.",
        2 * cm,
        y,
        size=10,
    )
    y -= 0.4 * cm
    write_line(f"Source des données : {source_txt}", 2 * cm, y)
    y -= 0.6 * cm

    write_line("1. Contexte et périmètre", 2 * cm, y, size=13, bold=True)
    y -= 0.7 * cm
    write_line(f"- Nombre de vols analysés : {len(df_filtered):,}", 2.3 * cm, y)
    y -= 0.4 * cm

    if kpi_nb_comp is not None and kpi_nb_types is not None:
        write_line(
            f"- Réseau opéré par {kpi_nb_comp} compagnies et {kpi_nb_types} types d'avions.",
            2.3 * cm,
            y,
        )
        y -= 0.4 * cm

    if kpi_dist_mean is not None and kpi_dur_mean is not None:
        write_line(
            f"- Distance moyenne : {kpi_dist_mean:.0f} km ; durée moyenne : {kpi_dur_mean:.0f} min.",
            2.3 * cm,
            y,
        )
        y -= 0.4 * cm

    if kpi_delay_mean is not None:
        med_txt = f"(médiane {kpi_delay_med:.1f} min)" if kpi_delay_med is not None else ""
        write_line(
            f"- Retard moyen global : {kpi_delay_mean:.1f} min {med_txt}",
            2.3 * cm,
            y,
        )
        y -= 0.4 * cm

    if kpi_co2_sum is not None:
        write_line(
            f"- Empreinte carbone totale : {kpi_co2_sum:,.0f} kg de CO₂.",
            2.3 * cm,
            y,
        )
        y -= 0.6 * cm

    write_line("Filtres appliqués :", 2 * cm, y, bold=True)
    y -= 0.4 * cm
    if filters_text:
        for ftxt in filters_text:
            write_line(f"- {ftxt}", 2.3 * cm, y)
            y -= 0.4 * cm
    else:
        write_line("- Aucun filtre spécifique.", 2.3 * cm, y)
        y -= 0.4 * cm

    y -= 0.4 * cm
    if tendance_txt:
        y = write_paragraph(tendance_txt, 2 * cm, y)

    c.showPage()

    # -------- Page 2 : Performance & retards --------
    y = height - 2 * cm
    write_line("2. Performance opérationnelle & retards", 2 * cm, y, size=13, bold=True)
    y -= 0.8 * cm

    y = write_paragraph(
        "Cette section analyse l'évolution des retards dans le temps, puis compare la performance des différentes "
        "compagnies aériennes.",
        2 * cm,
        y,
    )
    y -= 0.4 * cm

    # Graph retard temporel
    if ts_png is not None:
        img = ImageReader(BytesIO(ts_png))
        img_w = width - 4 * cm
        img_h = 7 * cm
        c.drawImage(
            img,
            2 * cm,
            y - img_h,
            width=img_w,
            height=img_h,
            preserveAspectRatio=True,
            mask="auto",
        )
        y -= img_h + 0.5 * cm
        y = write_paragraph(
            "Figure 1 – Évolution du retard moyen. On observe ici la dynamique globale des retards jour après jour. "
            "Les pics peuvent refléter des périodes de forte charge (saisonnalité), des contraintes ATC ou des "
            "perturbations météo.",
            2 * cm,
            y,
        )
        y -= 0.4 * cm

    # Graph retard par compagnie
    if airline_png is not None:
        if y < 8 * cm:
            c.showPage()
            y = height - 2 * cm
        img = ImageReader(BytesIO(airline_png))
        img_w = width - 4 * cm
        img_h = 7 * cm
        c.drawImage(
            img,
            2 * cm,
            y - img_h,
            width=img_w,
            height=img_h,
            preserveAspectRatio=True,
            mask="auto",
        )
        y -= img_h + 0.5 * cm
        y = write_paragraph(
            "Figure 2 – Retard moyen par compagnie aérienne. Cette comparaison met en évidence les acteurs qui "
            "surperforment (retards faibles) et ceux qui sous-performent. Les compagnies significativement au-dessus "
            "de la moyenne doivent être ciblées en priorité dans les plans d'amélioration.",
            2 * cm,
            y,
        )

    c.showPage()

    # -------- Page 3 : Réseau & Environnement --------
    y = height - 2 * cm
    write_line("3. Analyse du réseau de routes", 2 * cm, y, size=13, bold=True)
    y -= 0.8 * cm

    if routes_png is not None:
        img = ImageReader(BytesIO(routes_png))
        img_w = width - 4 * cm
        img_h = 7 * cm
        c.drawImage(
            img,
            2 * cm,
            y - img_h,
            width=img_w,
            height=img_h,
            preserveAspectRatio=True,
            mask="auto",
        )
        y -= img_h + 0.5 * cm
        y = write_paragraph(
            "Figure 3 – Top routes par nombre de vols. Les routes les plus fréquentées concentrent une part importante "
            "du trafic et constituent des axes prioritaires pour l'optimisation opérationnelle. Un retard chronique "
            "sur ces routes a un impact disproportionné sur l'expérience client et les coûts.",
            2 * cm,
            y,
        )
    else:
        y = write_paragraph(
            "Les informations d'origines et destinations ne sont pas suffisantes pour construire un graphique de routes.",
            2 * cm,
            y,
        )

    if seg_txt:
        if y < 4 * cm:
            c.showPage()
            y = height - 2 * cm
        write_line("4. Retards selon la typologie de vols", 2 * cm, y, size=13, bold=True)
        y -= 0.7 * cm
        y = write_paragraph(seg_txt, 2 * cm, y)

    if env_txt:
        if y < 4 * cm:
            c.showPage()
            y = height - 2 * cm
        write_line("5. Impact environnemental", 2 * cm, y, size=13, bold=True)
        y -= 0.7 * cm
        y = write_paragraph(env_txt, 2 * cm, y)

    c.showPage()

    # -------- Page 4 : Recommandations --------
    y = height - 2 * cm
    write_line("6. Recommandations stratégiques", 2 * cm, y, size=13, bold=True)
    y -= 0.8 * cm

    recos = [
        "Prioriser les compagnies présentant les retards moyens les plus élevés pour des revues opérationnelles ciblées "
        "(analyse des causes, ajustement des créneaux de départ, renforcement de la coordination au sol).",
        "Analyser les routes les plus critiques (fort volume et retard moyen élevé) afin d'identifier les contraintes "
        "récurrentes : saturation aéroportuaire, temps de rotation insuffisant, goulets d'étranglement dans la chaîne "
        "opérationnelle.",
        "Optimiser les vols long-courriers, particulièrement sensibles aux retards, en renforçant la préparation du vol, "
        "en ajustant les marges de planification et en surveillant les temps de roulage et d'attente au sol.",
        "Mettre en place un suivi régulier des indicateurs clés (retards, ponctualité, occupation, CO₂) via ce tableau "
        "de bord, avec comparaison avant/après actions correctives.",
        "Aligner la performance environnementale avec les décisions opérationnelles (choix appareil, plan de flotte, "
        "horaires) afin de réduire progressivement l'empreinte carbone sans dégrader la qualité de service.",
    ]

    for r in recos:
        y = write_paragraph("• " + r, 2 * cm, y)
        y -= 0.2 * cm
        if y < 3 * cm:
            c.showPage()
            y = height - 2 * cm

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


# ------------------------------------------------------------
# HEADER + RESUME + EXPORTS + ALERTES
# ------------------------------------------------------------
st.title("✈️ Aeronautics Operations & Fleet Dashboard")
st.caption(
    "Analyse interactive des performances de vols, des caractéristiques de flotte "
    "et du réseau de routes."
)

st.caption(
    f"Données filtrées : **{kpi_total_vols:,}** lignes sur **{len(df):,}** "
    f"({kpi_total_vols / max(len(df),1):.0%})."
)

# Chips de filtres actifs
active_filters = []
if selected_airline != "Toutes":
    active_filters.append(f"Compagnie : {selected_airline}")
if selected_aircraft != "Tous":
    active_filters.append(f"Type d’avion : {selected_aircraft}")
if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
    active_filters.append(f"Période : {date_range[0]} → {date_range[1]}")
if dist_min is not None and dist_max is not None:
    active_filters.append(f"Distance : {dist_min:.0f}–{dist_max:.0f} km")
if dur_min is not None and dur_max is not None:
    active_filters.append(f"Durée : {dur_min:.0f}–{dur_max:.0f} min")

chips_html = "".join(
    f'<span class="filter-chip">{txt}</span>' for txt in active_filters
)
if chips_html:
    st.markdown(chips_html, unsafe_allow_html=True)
else:
    st.markdown("_Aucun filtre spécifique appliqué._")

# Alertes simples (retard / CO2)
if kpi_delay_mean is not None and kpi_delay_mean > 15:
    st.error(
        f"⚠️ Niveau de retard critique : {kpi_delay_mean:.1f} min en moyenne sur la période filtrée."
    )
elif kpi_delay_mean is not None and kpi_delay_mean > 8:
    st.warning(
        f"⚠️ Niveau de retard significatif : {kpi_delay_mean:.1f} min en moyenne."
    )

if kpi_co2_sum is not None and kpi_co2_sum > 1_000_000:
    st.warning(
        f"🌍 Empreinte CO₂ très élevée sur la période : {kpi_co2_sum:,.0f} kg. "
        "Des actions de réduction sont à envisager."
    )

# Bloc export
st.markdown(
    """
    <div class="export-card">
        <div class="export-title">Exports & rapports</div>
    """,
    unsafe_allow_html=True,
)
exp_col1, exp_col2 = st.columns(2)
with exp_col1:
    st.write("**Données brutes filtrées**")
    st.caption("Exporter les lignes filtrées pour analyse dans Excel ou Power BI.")
    csv_bytes = fdf.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Exporter en CSV / Excel",
        data=csv_bytes,
        file_name="flights_filtered.csv",
        mime="text/csv",
        key="export_csv",
    )
with exp_col2:
    st.write("**Rapport PDF détaillé**")
    st.caption(
        "Rapport structuré avec synthèse, analyses graphiques et recommandations stratégiques."
    )
    if st.button("📄 Télécharger le rapport PDF", key="export_pdf"):
        try:
            pdf_bytes = generate_pdf_report(fdf, active_filters)
            st.download_button(
                "⬇️ Télécharger le PDF",
                data=pdf_bytes,
                file_name="rapport_aero_analytics.pdf",
                mime="application/pdf",
                key="download_pdf",
            )
        except ImportError:
            st.info(
                "Pour activer la génération du rapport PDF, ajoute `reportlab` et `kaleido` "
                "dans requirements.txt."
            )
        except Exception as e:
            st.error(
                "La génération du PDF n'est pas disponible dans cet environnement. "
                "Sur Streamlit Cloud, Kaleido peut nécessiter Google Chrome. "
                f"Détail technique : {e}"
            )

st.markdown("</div>", unsafe_allow_html=True)


st.divider()

# ------------------------------------------------------------
# KPI CARDS
# ------------------------------------------------------------

# KPI CARDS
# ------------------------------------------------------------
def metric_card(label: str, value: str, sub: str = ""):
    html = (
        '<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-sub">{sub}</div>'
        '</div>'
    )

    st.markdown(html, unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
with k1:
    metric_card(
        "Nombre de vols",
        f"{kpi_total_vols:,}",
        (
            f"{kpi_nb_comp} compagnies / {kpi_nb_types} types d’avions"
            if (kpi_nb_comp is not None and kpi_nb_types is not None)
            else ""
        ),
    )
with k2:
    if kpi_delay_mean is not None:
        metric_card(
            "Retard moyen",
            f"{kpi_delay_mean:.1f} min",
            f"Médiane : {kpi_delay_med:.1f} min" if kpi_delay_med is not None else "",
        )
with k3:
    if kpi_fuel_mean is not None:
        metric_card(
            "Fuel burn moyen",
            f"{kpi_fuel_mean:.0f} kg/vol",
            f"Total : {kpi_fuel_sum:,.0f} kg" if kpi_fuel_sum is not None else "",
        )
with k4:
    if kpi_co2_sum is not None:
        metric_card(
            "Empreinte CO₂",
            f"{kpi_co2_sum:,.0f} kg",
            f"{kpi_co2_mean:.0f} kg/vol" if kpi_co2_mean is not None else "",
        )

st.divider()

# ------------------------------------------------------------
# MODE COMPACT vs DÉTAILLÉ
# ------------------------------------------------------------
if view_mode == "Vue compacte":
    # Vue courte : quelques graphes essentiels uniquement
    st.subheader("Vue synthétique")

    col1, col2 = st.columns(2)

    # 1. Évolution du retard moyen
    with col1:
        st.markdown("#### Évolution du retard moyen")
        if {"date", "delay_min"}.issubset(fdf.columns):
            ts = (
                fdf.groupby("date", as_index=False)["delay_min"]
                .mean()
                .sort_values("date")
            )
            fig_ts = px.line(
                ts,
                x="date",
                y="delay_min",
                labels={"date": "Date", "delay_min": pretty("delay_min")},
            )
            fig_ts.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_ts, use_container_width=True)
        else:
            st.info("Les colonnes `date` et `delay_min` sont nécessaires pour cette vue.")

    # 2. Retard moyen par compagnie
    with col2:
        st.markdown("#### Retard moyen par compagnie")
        if {"airline", "delay_min"}.issubset(fdf.columns):
            agg = (
                fdf.groupby("airline", as_index=False)["delay_min"]
                .mean()
                .sort_values("delay_min", ascending=False)
            )
            fig_bar = px.bar(
                agg,
                x="airline",
                y="delay_min",
                labels={"airline": pretty("airline"), "delay_min": pretty("delay_min")},
            )
            fig_bar.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=40, b=80),
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Colonnes `airline` ou `delay_min` manquantes.")

    st.markdown("---")

    # 3. Top routes
    st.markdown("#### Top routes par nombre de vols")
    if {"origin", "destination"}.issubset(fdf.columns):
        fdf["route"] = fdf["origin"] + " → " + fdf["destination"]
        simple_counts = (
            fdf["route"]
            .value_counts()
            .head(10)
            .reset_index(name="vols_count")
            .rename(columns={"index": "route"})
        )
        fig_routes = px.bar(
            simple_counts,
            x="vols_count",
            y="route",
            orientation="h",
            labels={"vols_count": "Nombre de vols", "route": pretty("route")},
        )
        fig_routes.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_routes, use_container_width=True)
    else:
        st.info("Les colonnes `origin` et `destination` sont nécessaires pour cette vue.")

    st.info(
        "💡 Pour voir tous les onglets détaillés (insights, cartes, comparaisons, etc.), "
        "passe en **Vue détaillée** dans la barre de gauche."
    )

else:
    # --------------------------------------------------------
    # TABS (vue détaillée complète)
    # --------------------------------------------------------
    (
        tab_perf,
        tab_airline,
        tab_routes,
        tab_geo,
        tab_insights,
        tab_compare,
        tab_recog,
    ) = st.tabs(
        [
            "📈 Performance",
            "🏢 Compagnies & avions",
            "🛫 Routes",
            "🗺️ Carte & réseau",
            "💡 Insights automatiques",
            "🆚 Comparaison de scénarios",
            "🔍 Reconnaissance d’avion",
        ]
    )

    # =======================================================
    # TAB 1 : PERFORMANCE
    # =======================================================
    with tab_perf:
        col_ts, col_corr = st.columns((2, 1))

        # Série temporelle retard
        with col_ts:
            st.subheader("Évolution du retard moyen")
            if {"date", "delay_min"}.issubset(fdf.columns):
                ts = (
                    fdf.groupby("date", as_index=False)["delay_min"]
                    .mean()
                    .sort_values("date")
                )
                fig_ts = px.line(
                    ts,
                    x="date",
                    y="delay_min",
                    labels={"date": "Date", "delay_min": pretty("delay_min")},
                )
                fig_ts.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_ts, use_container_width=True)
            else:
                st.info("Les colonnes `date` et `delay_min` sont nécessaires pour cette vue.")

        # Heatmap corrélations
        with col_corr:
            st.subheader("Corrélations principales")
            if len(NUM_COLS) >= 2:
                corr = fdf[NUM_COLS].corr(numeric_only=True)
                corr_disp = corr.rename(index=pretty, columns=pretty)
                fig_corr = px.imshow(
                    corr_disp,
                    text_auto=False,
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                )
                fig_corr.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Pas assez de variables numériques pour la corrélation.")

        st.markdown("------")

        # Carte axes pro
        st.markdown(
            '<div class="axis-card">'
            '<div class="axis-title">Paramètres du nuage de points</div>',
            unsafe_allow_html=True,
        )

        if len(NUM_COLS) >= 2:
            c1, c2, c3 = st.columns([2.2, 2.2, 1.2])
            x_var = c1.selectbox(
                "Axe X",
                options=NUM_COLS,
                index=0,
                format_func=pretty,
            )
            y_var = c2.selectbox(
                "Axe Y",
                options=NUM_COLS,
                index=1,
                format_func=pretty,
            )
            color_options = ["Aucune"] + [
                c for c in ["airline", "aircraft_type"] if c in fdf.columns
            ]
            color_var = c3.selectbox(
                "Couleur",
                options=color_options,
                format_func=lambda x: "Aucune" if x == "Aucune" else pretty(x),
            )
        else:
            x_var = y_var = color_var = None

        st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("Relation entre deux paramètres")
        st.caption("Analyse fine de la relation entre deux variables du vol.")

        if x_var and y_var:
            labels = {x_var: pretty(x_var), y_var: pretty(y_var)}
            if color_var not in (None, "Aucune"):
                labels[color_var] = pretty(color_var)
            fig_sc = px.scatter(
                fdf,
                x=x_var,
                y=y_var,
                color=None if color_var == "Aucune" else color_var,
                hover_data=[
                    c
                    for c in ["airline", "aircraft_type", "origin", "destination"]
                    if c in fdf.columns
                ],
                labels=labels,
            )
            fig_sc.update_traces(marker=dict(size=7, opacity=0.85))
            fig_sc.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_sc, use_container_width=True)

            # petit résumé stats
            if x_var in fdf.columns and y_var in fdf.columns:
                subdf = fdf[[x_var, y_var]].dropna()
                if len(subdf) > 1:
                    corr_xy = subdf.corr().iloc[0, 1]
                    s1, s2 = st.columns(2)
                    with s1:
                        st.metric(
                            f"Corrélation {pretty(x_var)} / {pretty(y_var)}",
                            f"{corr_xy:.2f}",
                        )
                    with s2:
                        st.metric("Nombre de points", f"{len(subdf):,}")
        else:
            st.info("Choisis au moins deux variables numériques pour tracer le graphique.")

        st.markdown("------")

        # Distributions
        st.subheader("Distributions des paramètres clés")
        d1, d2, d3 = st.columns(3)

        if "distance_km" in fdf.columns:
            with d1:
                st.caption("Répartition des distances de vol.")
                fig_dist = px.histogram(
                    fdf,
                    x="distance_km",
                    nbins=30,
                    labels={"distance_km": pretty("distance_km")},
                )
                st.plotly_chart(fig_dist, use_container_width=True)

        if "duration_min" in fdf.columns:
            with d2:
                st.caption("Répartition des durées de vol.")
                fig_dur = px.histogram(
                    fdf,
                    x="duration_min",
                    nbins=30,
                    labels={"duration_min": pretty("duration_min")},
                )
                st.plotly_chart(fig_dur, use_container_width=True)

        if "delay_min" in fdf.columns:
            with d3:
                st.caption("Répartition des retards.")
                fig_del = px.histogram(
                    fdf,
                    x="delay_min",
                    nbins=30,
                    labels={"delay_min": pretty("delay_min")},
                )
                st.plotly_chart(fig_del, use_container_width=True)

    # =======================================================
    # TAB 2 : COMPAGNIES & AVIONS
    # =======================================================
    with tab_airline:
        c_left, c_right = st.columns(2)

        with c_left:
            st.subheader("Retard moyen par compagnie")
            if {"airline", "delay_min"}.issubset(fdf.columns):
                agg = (
                    fdf.groupby("airline", as_index=False)["delay_min"]
                    .mean()
                    .sort_values("delay_min", ascending=False)
                )
                fig_bar_airline = px.bar(
                    agg,
                    x="airline",
                    y="delay_min",
                    labels={
                        "airline": pretty("airline"),
                        "delay_min": pretty("delay_min"),
                    },
                )
                fig_bar_airline.update_layout(
                    xaxis_tickangle=-45,
                    height=420,
                    margin=dict(l=10, r=10, t=40, b=80),
                )
                st.plotly_chart(fig_bar_airline, use_container_width=True)
            else:
                st.info("Colonnes `airline` ou `delay_min` manquantes.")

        with c_right:
            st.subheader("Répartition des vols par type d’avion")
            if "aircraft_type" in fdf.columns:
                ac_counts = (
                    fdf.groupby("aircraft_type")
                    .size()
                    .reset_index(name="vols_count")
                )
                fig_pie_ac = px.pie(
                    ac_counts,
                    names="aircraft_type",
                    values="vols_count",
                    hole=0.35,
                    labels={
                        "aircraft_type": pretty("aircraft_type"),
                        "vols_count": "Nombre de vols",
                    },
                )
                fig_pie_ac.update_layout(
                    height=420,
                    margin=dict(l=10, r=10, t=40, b=10),
                    showlegend=True,
                )
                st.plotly_chart(fig_pie_ac, use_container_width=True)
            else:
                st.info("Colonne `aircraft_type` manquante.")

        st.markdown("------")

        st.subheader("Tableau de bord par compagnie")
        if {"airline", "delay_min"}.issubset(fdf.columns):
            cols_for_group = [c for c in NUM_COLS if c in fdf.columns]
            grouped = fdf.groupby("airline")[cols_for_group].agg(
                vols=("delay_min", "count"),
                delay_moyen=("delay_min", "mean"),
                distance_moy=("distance_km", "mean")
                if "distance_km" in fdf.columns
                else ("delay_min", "mean"),
            )
            st.dataframe(grouped.sort_values("delay_moyen", ascending=False))
        else:
            st.info("Impossible de construire le tableau de synthèse (colonnes manquantes).")

    # =======================================================
    # TAB 3 : ROUTES
    # =======================================================
    with tab_routes:
        st.subheader("Top routes par nombre de vols")
        if {"origin", "destination"}.issubset(fdf.columns):
            fdf["route"] = fdf["origin"] + " → " + fdf["destination"]
            simple_counts = (
                fdf["route"]
                .value_counts()
                .head(15)
                .reset_index(name="vols_count")
                .rename(columns={"index": "route"})
            )
            fig_routes = px.bar(
                simple_counts,
                x="vols_count",
                y="route",
                orientation="h",
                labels={"vols_count": "Nombre de vols", "route": pretty("route")},
            )
            fig_routes.update_layout(
                height=500,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_routes, use_container_width=True)
        else:
            st.info("Les colonnes `origin` et `destination` sont nécessaires pour cette vue.")

    # =======================================================
    # TAB 4 : CARTE & PYDECK
    # =======================================================
    with tab_geo:
        c_geo1, c_geo2 = st.columns(2)

        with c_geo1:
            st.subheader("Points géographiques des vols")
            if {"lat", "lon"}.issubset(fdf.columns):
                st.caption("Chaque point représente une observation de vol.")
                st.map(
                    fdf[["lat", "lon"]]
                    .dropna()
                    .rename(columns={"lat": "latitude", "lon": "longitude"})
                )
            else:
                st.info("Colonnes `lat` / `lon` manquantes pour afficher la carte.")

        with c_geo2:
            st.subheader("Flux de routes (PyDeck ArcLayer)")
            airport_coords = {
                "CMN": (33.3675, -7.5897),
                "RAK": (31.6069, -8.0363),
                "TNG": (35.7269, -5.9169),
                "AGA": (30.3250, -9.4131),
                "ORY": (48.7233, 2.3794),
                "MAD": (40.4722, -3.5608),
                "LHR": (51.4700, -0.4543),
                "IST": (41.2603, 28.7420),
            }

            if {"origin", "destination"}.issubset(fdf.columns):
                agg = (
                    fdf.groupby(["origin", "destination"], as_index=False)
                    .size()
                    .rename(columns={"size": "count"})
                )

                def add_coords(row):
                    o = airport_coords.get(row["origin"])
                    d = airport_coords.get(row["destination"])
                    if o and d:
                        return pd.Series(
                            {
                                "o_lat": o[0],
                                "o_lon": o[1],
                                "d_lat": d[0],
                                "d_lon": d[1],
                            }
                        )
                    return pd.Series(
                        {
                            "o_lat": np.nan,
                            "o_lon": np.nan,
                            "d_lat": np.nan,
                            "d_lon": np.nan,
                        }
                    )

                coords = agg.apply(add_coords, axis=1)
                arcs = pd.concat([agg, coords], axis=1).dropna()

                if not arcs.empty:
                    arc_layer = pdk.Layer(
                        "ArcLayer",
                        data=arcs,
                        get_source_position=["o_lon", "o_lat"],
                        get_target_position=["d_lon", "d_lat"],
                        get_source_color=[56, 189, 248],
                        get_target_color=[251, 191, 36],
                        get_width=2,
                    )
                    center_lat = float(
                        np.mean([airport_coords[k][0] for k in airport_coords])
                    )
                    center_lon = float(
                        np.mean([airport_coords[k][1] for k in airport_coords])
                    )
                    view_state = pdk.ViewState(
                        latitude=center_lat,
                        longitude=center_lon,
                        zoom=3,
                        pitch=0,
                    )
                    deck = pdk.Deck(
                        layers=[arc_layer],
                        initial_view_state=view_state,
                        tooltip={"text": "{origin} → {destination} ({count})"},
                    )
                    st.pydeck_chart(deck)
                else:
                    st.info("Aucun flux de route à afficher avec les filtres actuels.")
            else:
                st.info("Les colonnes `origin` et `destination` sont nécessaires pour PyDeck.")

    # =======================================================
    # TAB 5 : INSIGHTS AUTOMATIQUES + TOP PROBLÈMES
    # =======================================================
    with tab_insights:
        st.subheader("Synthèse automatique des données filtrées")

        st.markdown("### 1. Vue d’ensemble")
        st.write(f"- **Nombre de vols** : {kpi_total_vols:,}")
        if kpi_nb_comp is not None and kpi_nb_types is not None:
            st.write(
                f"- **Compagnies** : {kpi_nb_comp} – **Types d’avions** : {kpi_nb_types}"
            )
        if kpi_dist_mean is not None and kpi_dur_mean is not None:
            st.write(
                f"- **Distance moyenne** : {kpi_dist_mean:.0f} km – **Durée moyenne** : {kpi_dur_mean:.0f} min"
            )
        if kpi_delay_mean is not None:
            st.write(
                f"- **Retard moyen global** : {kpi_delay_mean:.1f} min "
                + (f"(médiane {kpi_delay_med:.1f} min)" if kpi_delay_med is not None else "")
            )

        st.markdown("### 2. Performance & retards")

        if {"airline", "delay_min"}.issubset(fdf.columns):
            grp = (
                fdf.groupby("airline")["delay_min"]
                .mean()
                .sort_values(ascending=False)
            )
            overall = fdf["delay_min"].mean()
            top_worst = grp.head(3)
            top_best = grp.tail(3)

            st.write("**Compagnies les plus en retard :**")
            for name, val in top_worst.items():
                st.write(f"- {name} : {val:.1f} min de retard moyen")

            st.write("**Compagnies les plus ponctuelles :**")
            for name, val in top_best.items():
                st.write(f"- {name} : {val:.1f} min de retard moyen")

            st.write(
                f"_Le retard moyen global est de {overall:.1f} minutes. "
                "Les compagnies significativement au-dessus de cette valeur doivent être priorisées._"
            )

        if {"distance_km", "delay_min"}.issubset(fdf.columns):
            short = fdf[fdf["distance_km"] <= 1000]["delay_min"].mean()
            medium = fdf[
                (fdf["distance_km"] > 1000) & (fdf["distance_km"] <= 3000)
            ]["delay_min"].mean()
            long = fdf[fdf["distance_km"] > 3000]["delay_min"].mean()

            st.markdown("**Retard moyen selon la typologie de vols :**")
            st.write(f"- Vols courts (≤ 1 000 km) : {short:.1f} min")
            st.write(f"- Vols moyens (1 000–3 000 km) : {medium:.1f} min")
            st.write(f"- Vols long-courriers (> 3 000 km) : {long:.1f} min")

        st.markdown("### 3. Top problèmes récurrents")

        if {"origin", "destination", "delay_min"}.issubset(fdf.columns):
            fdf["route"] = fdf["origin"] + " → " + fdf["destination"]
            route_delay = (
                fdf.groupby("route")["delay_min"]
                .mean()
                .sort_values(ascending=False)
            )
            route_vols = fdf["route"].value_counts()
            crit = (
                pd.DataFrame({"retard_moyen": route_delay})
                .join(route_vols.rename("nb_vols"))
                .dropna()
            )
            crit = crit[crit["nb_vols"] >= 5]
            if not crit.empty:
                top_crit = crit.head(5)
                st.write("**Routes les plus critiques (retard élevé & volume significatif) :**")
                st.dataframe(top_crit)
            else:
                st.write("Pas de route critique avec le filtre actuel.")
        else:
            st.write("Information de routes ou retards insuffisante pour calculer les problèmes récurrents.")

        st.markdown("### 4. Environnement")

        if {"co2_kg", "fuel_burn_kg"}.issubset(fdf.columns):
            st.write(
                f"- CO₂ total émis : **{fdf['co2_kg'].sum():,.0f} kg** – soit "
                f"**{fdf['co2_kg'].mean():.0f} kg** par vol en moyenne."
            )
            st.write(
                f"- Carburant total consommé : **{fdf['fuel_burn_kg'].sum():,.0f} kg**."
            )
        else:
            st.write("_Les indicateurs environnementaux ne sont pas disponibles._")

        st.markdown("### 5. Recommandations opérationnelles")

        st.write(
            "- **Prioriser les compagnies les plus en retard** pour des plans d’action ciblés "
            "(processus au sol, planification, gestion des créneaux…)."
        )
        st.write(
            "- **Analyser les routes les plus critiques** (retards élevés / forte fréquence) "
            "afin d’identifier les causes récurrentes : saturation aéroportuaire, ATC, rotation avion, etc."
        )
        st.write(
            "- **Surveiller plus finement les vols long-courriers**, où chaque minute de retard "
            "a un impact important sur les coûts opérationnels et l’empreinte carbone."
        )
        st.write(
            "- **Mettre en place un suivi régulier du tableau de bord** (hebdomadaire ou mensuel) "
            "et comparer systématiquement les indicateurs avant/après actions correctives."
        )

    # =======================================================
    # TAB 6 : COMPARAISON DE SCÉNARIOS (2 COMPAGNIES)
    # =======================================================
    with tab_compare:
        st.subheader("Comparaison de scénarios par compagnie")

        if "airline" not in fdf.columns:
            st.info("La colonne `airline` est nécessaire pour cette comparaison.")
        else:
            comp_list = sorted(fdf["airline"].dropna().unique().tolist())
            if len(comp_list) < 2:
                st.info("Il faut au moins 2 compagnies différentes dans les données filtrées.")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    comp_a = st.selectbox("Compagnie A", comp_list, index=0)
                with c2:
                    comp_b = st.selectbox("Compagnie B", comp_list, index=1)

                df_a = fdf[fdf["airline"] == comp_a]
                df_b = fdf[fdf["airline"] == comp_b]

                colA, colB = st.columns(2)
                for col, name, subdf in [(colA, comp_a, df_a), (colB, comp_b, df_b)]:
                    with col:
                        st.markdown(f"### {name}")
                        n_vols = len(subdf)
                        delay_mean = (
                            subdf["delay_min"].mean()
                            if "delay_min" in subdf.columns
                            else None
                        )
                        dist_mean = (
                            subdf["distance_km"].mean()
                            if "distance_km" in subdf.columns
                            else None
                        )
                        co2_mean = (
                            subdf["co2_kg"].mean()
                            if "co2_kg" in subdf.columns
                            else None
                        )
                        st.write(f"- Vols : **{n_vols:,}**")
                        if delay_mean is not None:
                            st.write(f"- Retard moyen : **{delay_mean:.1f} min**")
                        if dist_mean is not None:
                            st.write(f"- Distance moyenne : **{dist_mean:.0f} km**")
                        if co2_mean is not None:
                            st.write(f"- CO₂ moyen par vol : **{co2_mean:.0f} kg**")

                # Petit graphique comparatif
                if "delay_min" in fdf.columns:
                    comp_df = (
                        fdf[fdf["airline"].isin([comp_a, comp_b])]
                        .groupby("airline", as_index=False)["delay_min"]
                        .mean()
                    )
                    fig_cmp = px.bar(
                        comp_df,
                        x="airline",
                        y="delay_min",
                        labels={
                            "airline": "Compagnie",
                            "delay_min": "Retard moyen (min)",
                        },
                        title="Comparaison du retard moyen",
                    )
                    st.plotly_chart(fig_cmp, use_container_width=True)

    # =======================================================
    # TAB 7 : RECONNAISSANCE D’AVION
    # =======================================================
    with tab_recog:
        render_aircraft_recognition_tab()

# ------------------------------------------------------------
# DONNÉES BRUTES (OPTIONNEL)
# ------------------------------------------------------------
if show_raw:
    st.markdown("### 🧾 Données filtrées")
    st.dataframe(fdf.reset_index(drop=True))
