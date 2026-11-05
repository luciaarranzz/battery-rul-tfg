"""Estilo visual común de la aplicación: tarjetas, formularios y pantalla de acceso.

El tema base (colores y tipografía) está en .streamlit/config.toml; aquí solo se ajustan
los detalles que el tema no cubre. Los colores y la fuente propios están al principio del
CSS, como variables, para poder cambiarlos en un único sitio.
"""
import streamlit as st

CSS = """
<style>
  @import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap");

  :root {
    --fuente: "Inter", "Segoe UI", system-ui, sans-serif;
    --fondo: #16181D;
    --tarjeta: #1E2128;
    --borde: #2E323B;
    --campo: #14161B;
    --acento: #FFB020;
    --acento2: #FFB020;
    --texto: #ECEAE4;
    --apagado: #9A9CA3;
  }

  .block-container { padding-top: 2.5rem; max-width: 1200px; }

  /* Tipografía: se aplica a textos, no a los iconos de Streamlit */
  html, body, .stApp, p, label, button, input, textarea,
  h1, h2, h3, h4, h5, h6, li, td, th,
  [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
  [data-testid="stCaptionContainer"], [data-testid="stMarkdownContainer"] {
    font-family: var(--fuente) !important;
  }
  h1, h2, h3, .login-hero h1, .marca-lateral span { letter-spacing: -0.01em; font-weight: 700; }

  /* Tarjetas de indicadores */
  [data-testid="stMetric"] {
    background: linear-gradient(160deg, var(--tarjeta) 0%, #191C22 100%);
    border: 1px solid var(--borde);
    border-left: 4px solid var(--acento);
    border-radius: 12px;
    padding: 1rem 1.25rem;
  }
  [data-testid="stMetricLabel"] { color: var(--apagado); letter-spacing: .02em; }
  [data-testid="stMetricValue"] { font-weight: 700; }

  /* Formularios y desplegables como tarjetas */
  [data-testid="stForm"] {
    background: var(--tarjeta);
    border: 1px solid var(--borde);
    border-radius: 16px;
    padding: 1.5rem;
  }
  [data-testid="stExpander"] { border-radius: 12px; background: var(--tarjeta); }

  /* Campos de texto */
  [data-testid="stTextInputRootElement"],
  [data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: var(--campo) !important;
    border: 1px solid var(--borde) !important;
    border-radius: 10px;
  }
  [data-testid="stTextInputRootElement"]:focus-within { border-color: var(--acento) !important; }

  /* Botones principales: texto oscuro sobre el ámbar */
  [data-testid^="stBaseButton-primary"] p { color: #16181D !important; font-weight: 600; }

  /* Zona de subida de ficheros */
  [data-testid="stFileUploaderDropzone"] {
    background: var(--tarjeta);
    border: 1px dashed var(--acento2);
    border-radius: 12px;
  }

  /* Panel de marca de la pantalla de acceso */
  .login-hero {
    min-height: 430px;
    padding: 2.5rem 2rem;
    border-radius: 16px;
    border: 1px solid var(--borde);
    display: flex;
    flex-direction: column;
    justify-content: center;
    background:
      radial-gradient(circle at 15% 10%, rgba(255, 176, 32, .22), transparent 55%),
      repeating-linear-gradient(115deg, rgba(255, 255, 255, .035) 0 2px, transparent 2px 26px),
      linear-gradient(160deg, #1E2128 0%, #14161B 100%);
  }
  .login-hero svg { --contorno: var(--texto); --barras: var(--acento); }
  .login-hero h1 { margin: 1rem 0 .25rem; padding: 0; font-size: 2.6rem; line-height: 1.1; color: var(--acento) !important; }
  .login-hero .lema { color: #C9C7C0 !important; font-size: 1rem; margin: 0 0 1.5rem; }
  .login-hero ul { list-style: none; padding: 0; margin: 0; }
  .login-hero li { margin: .5rem 0; font-size: .92rem; color: var(--texto) !important; }
  .login-hero li::before { content: ""; display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; background: var(--acento); margin-right: .7rem; }
  .login-titulo { margin: 2.5rem 0 1rem; }
  .login-titulo h2 { margin: 0; padding: 0; font-size: 1.7rem; }
  .login-titulo p { color: var(--apagado); margin: .25rem 0 0; }

  /* Marca en la barra lateral */
  .marca-lateral { display: flex; align-items: center; gap: .6rem; margin-bottom: 1rem; }
  .marca-lateral svg { --contorno: var(--texto); --barras: var(--acento); }
  .marca-lateral span { font-size: 1.25rem; font-weight: 700; }
  .marca-lateral span b { color: var(--acento); }
</style>
"""

LOGO_BATERIA = """
<svg width="96" height="48" viewBox="0 0 96 48" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Batería">
  <rect x="2" y="4" width="84" height="40" rx="8" fill="none" style="stroke: var(--contorno)" stroke-width="3"/>
  <rect x="88" y="16" width="6" height="16" rx="2" style="fill: var(--contorno)"/>
  <rect x="9" y="11" width="16" height="26" rx="3" style="fill: var(--barras)"/>
  <rect x="29" y="11" width="16" height="26" rx="3" style="fill: var(--barras)"/>
  <rect x="49" y="11" width="16" height="26" rx="3" style="fill: var(--barras)"/>
  <rect x="69" y="11" width="12" height="26" rx="3" style="fill: var(--barras)" opacity=".35"/>
</svg>
"""

LOGO_PEQUENO = LOGO_BATERIA.replace('width="96" height="48"', 'width="48" height="24"')

# El HTML va sin sangrar: con 4 espacios o más, Markdown lo trataría como código.
HERO = f"""<div class="login-hero">
{LOGO_BATERIA}
<h1>Battery RUL</h1>
<p class="lema">Conoce la salud de la batería de tu vehículo.</p>
<ul>
<li>Estado de salud (SOH) estimado ciclo a ciclo</li>
<li>Vida útil restante (RUL) hasta el fin de vida</li>
<li>Sube las lecturas de tu BMS en formato CSV</li>
</ul>
</div>"""


def aplicar():
    """Inserta el CSS común. Hay que llamarlo una vez por ejecución del script."""
    st.markdown(CSS, unsafe_allow_html=True)


def hero_login():
    """Panel izquierdo de la pantalla de acceso: marca y qué hace la aplicación."""
    st.markdown(HERO, unsafe_allow_html=True)


def titulo_login():
    st.markdown(
        '<div class="login-titulo"><h2>Inicia sesión</h2>'
        "<p>Accede con tu cuenta para ver tu vehículo.</p></div>",
        unsafe_allow_html=True,
    )


def marca_lateral():
    """Logo y nombre de la aplicación en la parte superior de la barra lateral."""
    st.markdown(
        f'<div class="marca-lateral">{LOGO_PEQUENO}<span>Battery <b>RUL</b></span></div>',
        unsafe_allow_html=True,
    )