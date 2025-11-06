# Importation des bibliothèques nécessaires
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Configuration de la page Streamlit
st.set_page_config(
    page_title="NIAKO Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.gabar-analytics.com',
        'Report a bug': 'https://www.gabar-analytics.com/support',
        'About': "Plateforme d'Intelligence Client pour Institutions Financières"
    }
)

# Style CSS personnalisé amélioré
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    :root {
        --primary-blue: #2563eb;
        --secondary-blue: #3b82f6;
        --dark-bg: #f8fafc;
        --card-bg: #ffffff;
        --card-hover: #f1f5f9;
        --text-primary: #0f172a;
        --text-secondary: #64748b;
        --accent-green: #10b981;
        --accent-red: #ef4444;
        --accent-yellow: #f59e0b;
        --accent-purple: #8b5cf6;
        --border-color: #e2e8f0;
        --shadow: rgba(15, 23, 42, 0.08);
    }
    
    * {
        font-family: 'Inter', sans-serif;
        box-sizing: border-box;
    }
    
    .main {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        color: var(--text-primary);
        padding-bottom: 60px;
    }
    
    /* En-tête avec animation */
    .banking-header {
        background: linear-gradient(135deg, #1e40af 0%, #2563eb 50%, #3b82f6 100%);
        padding: 3rem 2rem;
        border-radius: 0 0 30px 30px;
        margin: -1rem -1rem 2rem -1rem;
        box-shadow: 0 10px 40px rgba(37, 99, 235, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .banking-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%);
        animation: pulse 8s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1) rotate(0deg); opacity: 0.5; }
        50% { transform: scale(1.1) rotate(180deg); opacity: 0.8; }
    }
    
    .banking-header h1 {
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        position: relative;
        z-index: 1;
        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
        color: white;
    }
    
    .banking-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.95;
        position: relative;
        z-index: 1;
        font-weight: 400;
        color: white;
    }
    
    /* Cartes métriques uniformes */
    .metric-card {
        background: var(--card-bg);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 20px var(--shadow);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        transition: opacity 0.4s;
    }
    
    .metric-card.card-blue::before { background: linear-gradient(90deg, #2563eb, #3b82f6); }
    .metric-card.card-red::before { background: linear-gradient(90deg, #dc2626, #ef4444); }
    .metric-card.card-green::before { background: linear-gradient(90deg, #059669, #10b981); }
    .metric-card.card-yellow::before { background: linear-gradient(90deg, #d97706, #f59e0b); }
    
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px var(--shadow);
        border-color: var(--primary-blue);
    }
    
    .metric-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
    }
    
    .metric-title {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.7;
    }
    
    .metric-title.title-blue { color: #2563eb; }
    .metric-title.title-red { color: #dc2626; }
    .metric-title.title-green { color: #059669; }
    .metric-title.title-yellow { color: #d97706; }
    
    .metric-icon {
        font-size: 2rem;
        opacity: 0.3;
    }
    
    .metric-value {
        font-size: 2.75rem;
        font-weight: 800;
        color: var(--text-primary);
        line-height: 1;
        margin: 0.5rem 0;
    }
    
    .metric-subtitle {
        font-size: 0.875rem;
        color: var(--text-secondary);
        font-weight: 500;
    }
    
    /* Sidebar améliorée */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-right: 1px solid var(--border-color);
    }
    
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }
    
    /* Boutons et selectbox */
    .stSelectbox, .stSlider {
        margin-bottom: 1.5rem;
    }
    
    div[data-baseweb="select"] {
        background-color: var(--card-bg);
        border-color: var(--border-color);
        border-radius: 8px;
    }
    
    /* Container pour les graphiques */
    .chart-container {
        background: var(--card-bg);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 20px var(--shadow);
        margin-bottom: 1.5rem;
    }
    
    /* Dataframe */
    .dataframe-container {
        background: var(--card-bg);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 20px var(--shadow);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: var(--card-bg);
        border-radius: 8px;
        border: 1px solid var(--border-color);
        font-weight: 600;
        color: var(--text-primary);
    }
    
    /* Sections */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-color), transparent);
        margin: 2.5rem 0;
    }
    
    /* Amélioration des titres */
    h4 {
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid var(--primary-blue);
        display: inline-block;
    }
    
    /* Fix pour les colonnes Streamlit */
    [data-testid="column"] {
        padding: 0 0.5rem;
    }
    
    /* Uniformisation de la hauteur des graphiques */
    .js-plotly-plot {
        height: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='banking-header' style='text-align: center;'>
    <h1>🏦 NIAKO Analytics</h1>
    <p>Intelligence Client 360° – Analyse Prédictive, Crédit, Risques & Finance</p>
</div>
""", unsafe_allow_html=True)

# Chargement des données
@st.cache_data
def charger_donnees():
    try:
        df = pd.read_csv("BankChurners.csv")
        colonnes_requises = [
            'Attrition_Flag', 'Customer_Age', 'Credit_Limit', 'Total_Trans_Amt',
            'Total_Relationship_Count', 'Months_on_book', 'Avg_Utilization_Ratio',
            'Total_Revolving_Bal', 'Total_Trans_Ct', 'Education_Level'
        ]
        colonnes_manquantes = [col for col in colonnes_requises if col not in df.columns]
        if colonnes_manquantes:
            st.error(f"Colonnes manquantes : {', '.join(colonnes_manquantes)}")
            return None
        df['Attrition_Flag'] = df['Attrition_Flag'].map({'Attrited Customer': 1, 'Existing Customer': 0})
        df['CLV'] = (df['Total_Trans_Amt'] * df['Total_Relationship_Count']) / df['Months_on_book'].replace(0, 1)
        df['Score_Risque'] = np.where(df['Avg_Utilization_Ratio'] > 0.75, 3,
                                    np.where(df['Avg_Utilization_Ratio'] > 0.5, 2, 1))
        return df.dropna(subset=colonnes_requises)
    except Exception as e:
        st.error(f"Erreur de chargement : {str(e)}")
        return None

# Modèle prédictif
@st.cache_resource
def entrainer_modele_churn(df):
    try:
        features = ['Customer_Age', 'Credit_Limit', 'Total_Revolving_Bal',
                   'Avg_Utilization_Ratio', 'Total_Trans_Ct', 'CLV']
        X = df[features]
        y = df['Attrition_Flag']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        df['Probabilite_Churn'] = model.predict_proba(X)[:, 1]
        return df, model.feature_importances_
    except Exception as e:
        st.error(f"Erreur d'entraînement : {str(e)}")
        return df, None

# PAGE : Dashboard Global
def page_dashboard(df, importance_caracteristiques, type_segmentation):
    # Section KPI avec design uniforme
    cols = st.columns(4)
    kpis = [
        ("Portefeuille Client", "10,127", "Clients Actifs", "card-blue", "title-blue", "👥"),
        ("Taux de Churn", "16.1%", "vs trim précédent", "card-red", "title-red", "📉"),
        ("CLV Médian", "$366", "Valeur Client", "card-green", "title-green", "💰"),
        ("Exposition Risque", "747", "Clients à Haut Risque", "card-yellow", "title-yellow", "⚠️")
    ]

    for col, (titre, valeur, sous_titre, card_class, title_class, icon) in zip(cols, kpis):
        with col:
            st.markdown(f"""
                <div class='metric-card {card_class}'>
                    <div class='metric-header'>
                        <div class='metric-title {title_class}'>{titre}</div>
                        <div class='metric-icon'>{icon}</div>
                    </div>
                    <div class='metric-value'>{valeur}</div>
                    <div class='metric-subtitle'>{sous_titre}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Visualisations principales
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### 📊 Distribution du Risque de Churn")
        
        fig = px.histogram(
            df, 
            x='Probabilite_Churn', 
            nbins=50,
            color_discrete_sequence=['#3b82f6']
        )
        fig.update_layout(
            template="plotly_white",
            bargap=0.1,
            xaxis_title="Probabilité de Churn",
            yaxis_title="Nombre de Clients",
            paper_bgcolor='rgba(255, 255, 255, 1)',
            plot_bgcolor='rgba(248, 250, 252, 0.5)',
            font=dict(family="Inter", size=12, color='#0f172a'),
            hoverlabel=dict(bgcolor="white", font_size=13),
            height=400,
            margin=dict(t=20, b=60, l=60, r=20)
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0')

        
        st.plotly_chart(fig, use_container_width=True, key="hist_churn")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Facteurs d'Influence")
        
        if importance_caracteristiques is not None:
            features = ['Âge', 'Limite Crédit', 'Solde Récurrent', 
                        'Utilisation', 'Transactions', 'CLV']
            df_imp = pd.DataFrame({
                'Facteur': features,
                'Importance': importance_caracteristiques
            }).sort_values('Importance', ascending=True)
            
            fig = go.Figure(go.Bar(
                x=df_imp['Importance'],
                y=df_imp['Facteur'],
                orientation='h',
                marker=dict(
                    color=df_imp['Importance'],
                    colorscale=[[0, '#dbeafe'], [1, '#2563eb']],
                    line=dict(color='#1e40af', width=1)
                ),
                text=[f'{x:.1%}' for x in df_imp['Importance']],
                textposition='outside',
                textfont=dict(size=11, color='#0f172a')
            ))
            
            fig.update_layout(
                template="plotly_white",
                xaxis_title="Importance Relative",
                yaxis_title="",
                paper_bgcolor='rgba(255, 255, 255, 1)',
                plot_bgcolor='rgba(248, 250, 252, 0.5)',
                font=dict(family="Inter", size=12, color='#0f172a'),
                showlegend=False,
                height=400,
                margin=dict(t=20, b=60, l=120, r=80)
            )
          

            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0')

            st.plotly_chart(fig, use_container_width=True, key="bar_importance")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Segmentation
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### 🎨 Segmentation Clientèle")
    page_segments(df, type_segmentation)

# PAGE : Segmentation
def page_segments(df, type_segmentation):
    segmentation_map = {
        "Comportementale": ['Total_Trans_Ct', 'Total_Trans_Amt', 'Total_Revolving_Bal'],
        "Valeur Client": ['CLV', 'Customer_Age', 'Months_on_book'],
        "Risque Crédit": ['Credit_Limit', 'Avg_Utilization_Ratio', 'Score_Risque']
    }
    
    cols_seg = segmentation_map[type_segmentation]
    n_clusters = st.slider("Nombre de Segments", 2, 5, 3, key="n_segments")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Segment'] = kmeans.fit_predict(df[cols_seg])
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    colors = ['#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']
    
    fig = px.scatter(
        df, 
        x=cols_seg[0], 
        y=cols_seg[1], 
        color='Segment',
        size=cols_seg[2] if len(cols_seg) > 2 else None,
        template="plotly_white",
        labels={cols_seg[0]: cols_seg[0].replace('_', ' '),
                cols_seg[1]: cols_seg[1].replace('_', ' ')},
        color_discrete_sequence=colors
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(255, 255, 255, 1)',
        plot_bgcolor='rgba(248, 250, 252, 0.5)',
        font=dict(family="Inter", size=12, color='#0f172a'),
        height=500,
        margin=dict(t=20, b=60, l=60, r=20)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0')

    
    st.plotly_chart(fig, use_container_width=True, key="scatter_segments")
    st.markdown('</div>', unsafe_allow_html=True)

# PAGE : Prédiction Churn
def page_prediction(df):
    st.markdown("## 🔮 Prédiction de Churn Client")
    
    if 'Probabilite_Churn' not in df.columns:
        st.error("❌ Problème de chargement des prédictions")
        return

    col_data, col_viz = st.columns([1, 1], gap="large")

    with col_data:
        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
        st.markdown("### 📋 Clients à Risque Élevé")
        
        df_display = df[['Customer_Age', 'Credit_Limit', 'Total_Trans_Ct', 
                        'CLV', 'Probabilite_Churn']].sort_values('Probabilite_Churn', ascending=False).head(100)
        
        st.dataframe(
            df_display.style.format({
                'CLV': '${:,.0f}',
                'Probabilite_Churn': '{:.1%}',
                'Credit_Limit': '${:,.0f}'
            }).background_gradient(subset=['Probabilite_Churn'], cmap='RdYlGn_r'),
            height=550,
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_viz:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 📊 Analyse Multidimensionnelle")
        
        fig = px.scatter(
            df,
            x='Customer_Age',
            y='CLV',
            size='Credit_Limit',
            color='Probabilite_Churn',
            hover_data={
                'Total_Trans_Amt': ':.2f',
                'Avg_Utilization_Ratio': ':.0%',
                'Customer_Age': True,
                'CLV': ':.2f',
                'Probabilite_Churn': ':.1%'
            },
            color_continuous_scale='RdYlGn_r',
            template='plotly_white'
        )
        
        fig.update_layout(
            xaxis_title="Âge du Client (années)",
            yaxis_title="Customer Lifetime Value (USD)",
            coloraxis_colorbar_title='Probabilité<br>de Churn',
            paper_bgcolor='rgba(255, 255, 255, 1)',
            plot_bgcolor='rgba(248, 250, 252, 0.5)',
            font=dict(family="Inter", size=12, color='#0f172a'),
            height=550,
            margin=dict(t=20, b=60, l=60, r=20)
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0')

        
        st.plotly_chart(fig, use_container_width=True, key="scatter_prediction")
        st.markdown('</div>', unsafe_allow_html=True)

    # Section des filtres avancés
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🔎 Filtres Avancés", expanded=False):
        cols = st.columns(3)
        with cols[0]:
            min_prob = st.slider("Probabilité minimale", 0.0, 1.0, 0.0, 0.01, key="min_prob_filter")
        with cols[1]:
            max_prob = st.slider("Probabilité maximale", 0.0, 1.0, 1.0, 0.01, key="max_prob_filter")
        with cols[2]:
            filtered = df['Probabilite_Churn'].between(min_prob, max_prob).sum()
            st.metric("Clients filtrés", f"{filtered:,}", help="Clients dans la plage sélectionnée")

# Fonction principale
def main():
    df = charger_donnees()
    if df is None:
        return
    
    df, feature_importances = entrainer_modele_churn(df)
    
    # Configuration sidebar
    with st.sidebar:
        st.markdown("### 🎯 Navigation")
        page = st.selectbox("Menu Principal", ["Dashboard Global", "Prédiction Churn"], label_visibility="collapsed")
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        if page == "Dashboard Global":
            st.markdown("### ⚙️ Configuration")
            type_seg = st.selectbox("Type de Segmentation", 
                                   ["Comportementale", "Valeur Client", "Risque Crédit"])
        else:
            st.markdown("### ⚙️ Paramètres")
            seuil_churn = st.slider("Seuil d'alerte", 0.0, 1.0, 0.7, step=0.05, key="seuil_sidebar")
            df = df[df['Probabilite_Churn'] >= seuil_churn]
        
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 🎚️ Filtres Globaux")
        plage_clv = st.slider("Filtre CLV ($)", 
                             float(df['CLV'].min()), 
                             float(df['CLV'].max()), 
                             (float(df['CLV'].quantile(0.25)), float(df['CLV'].quantile(0.75))),
                             key="clv_filter")
        df = df[df['CLV'].between(plage_clv[0], plage_clv[1])]
        
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
            <div style='text-align: center; padding: 1rem; color: #64748b; font-size: 0.85rem;'>
                <p style='margin: 0.25rem 0;'><strong>© 2024 NIAKO Analytics</strong></p>
                <p style='margin: 0.25rem 0;'>Powered by Niako Analytics</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Gestion des pages
    if page == "Dashboard Global":
        page_dashboard(df, feature_importances, type_seg)
    else:
        page_prediction(df)

if __name__ == "__main__":
    main()