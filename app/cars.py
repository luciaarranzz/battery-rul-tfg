"""Siluetas animadas de vehículos (ilustrativas) para el dashboard."""

# Por vehículo: autonomía con batería nueva (supuesto editable), forma de la
# carrocería, ventanas, posición de faros y ruedas (x, y, radio).
CARS = {
    "Urbano": {
        "range_km": 250, "front": 362, "rear": 50, "head_y": 100, "tail_y": 92,
        "wheels": [(108, 114, 22), (290, 114, 22)],
        "body": "M50 114 L50 92 Q52 84 64 82 L110 76 Q124 54 160 50 L232 50 Q262 52 284 76 L332 84 Q360 88 362 104 L362 114 Z",
        "windows": [
            "M130 76 Q142 59 162 58 L196 58 L196 76 Z",
            "M206 58 L232 58 Q252 62 268 76 L206 76 Z",
        ],
    },
    "Berlina": {
        "range_km": 400, "front": 380, "rear": 22, "head_y": 100, "tail_y": 94,
        "wheels": [(98, 114, 22), (304, 114, 22)],
        "body": "M22 114 L22 98 Q22 90 36 87 L98 80 Q128 54 168 52 L240 52 Q272 54 296 80 L352 87 Q378 91 380 104 L380 114 Z",
        "windows": [
            "M118 80 Q140 61 170 60 L205 60 L205 80 Z",
            "M215 60 L240 60 Q262 63 280 80 L215 80 Z",
        ],
    },
    "SUV": {
        "range_km": 450, "front": 378, "rear": 24, "head_y": 98, "tail_y": 84,
        "wheels": [(100, 114, 25), (300, 114, 25)],
        "body": "M24 114 L24 84 Q24 74 36 72 L70 68 L92 44 Q98 38 108 38 L250 38 Q270 40 286 62 L346 76 Q376 82 378 100 L378 114 Z",
        "windows": [
            "M104 66 L118 47 L170 47 L170 66 Z",
            "M180 47 L248 47 Q262 50 274 66 L180 66 Z",
        ],
    },
    "Furgoneta": {
        "range_km": 250, "front": 380, "rear": 28, "head_y": 98, "tail_y": 70,
        "wheels": [(100, 114, 24), (306, 114, 24)],
        "body": "M28 114 L28 40 Q28 32 38 32 L270 32 Q296 34 312 58 L356 78 Q378 84 380 100 L380 114 Z",
        "windows": [
            "M276 40 Q292 42 303 58 L276 58 Z",
            "M190 40 L262 40 L262 58 L190 58 Z",
        ],
    },
}

# Los colores de cada tema (los mismos que .streamlit/config.toml).
PALETTES = {
    "light": {"bg": "#FFFFFF", "ink": "#1B1B1F", "muted": "#6B6B72", "accent": "#E8590C", "bad": "#D64545"},
    "dark": {"bg": "#16181D", "ink": "#ECEAE4", "muted": "#9A9CA3", "accent": "#FFB020", "bad": "#FF5A4E"},
}

STYLE = """
<style>
  body { margin: 0; }
  svg { width: 100%; max-width: 560px; height: auto; display: block; margin: 0 auto; }
  .bob { animation: bob .6s ease-in-out infinite; }
  @keyframes bob { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-1.4px); } }
  .spin { transform-box: fill-box; transform-origin: center; animation: spin .55s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .road { stroke: var(--muted); stroke-width: 2; stroke-dasharray: 22 16; animation: road .45s linear infinite; }
  @keyframes road { to { stroke-dashoffset: 38; } }
  @media (prefers-reduced-motion: reduce) { .bob, .spin, .road { animation: none; } }
</style>
"""


def wheel_svg(x, y, r):
    """Rueda: neumático, llanta y tres radios que giran."""
    s = r - 9
    h = s * 0.866
    return f"""
    <circle cx="{x}" cy="{y}" r="{r + 7}" fill="var(--bg)"/>
    <circle cx="{x}" cy="{y}" r="{r}" fill="var(--ink)"/>
    <circle cx="{x}" cy="{y}" r="{r - 6}" fill="var(--bg)"/>
    <g class="spin" stroke="var(--ink)" stroke-width="2.4" stroke-linecap="round">
      <line x1="{x - s}" y1="{y}" x2="{x + s}" y2="{y}"/>
      <line x1="{x - s / 2}" y1="{y - h}" x2="{x + s / 2}" y2="{y + h}"/>
      <line x1="{x - s / 2}" y1="{y + h}" x2="{x + s / 2}" y2="{y - h}"/>
    </g>
    <circle cx="{x}" cy="{y}" r="3.2" fill="var(--ink)"/>"""


def car_html(kind, theme="light"):
    """Devuelve el HTML animado del vehículo `kind` con los colores de `theme`."""
    car = CARS[kind]
    p = PALETTES[theme]
    wheels = "".join(wheel_svg(x, y, r) for x, y, r in car["wheels"])
    windows = "".join(
        f'<path d="{d}" fill="var(--bg)" opacity=".92"/>' for d in car["windows"]
    )
    return f"""{STYLE}
    <div style="--bg:{p['bg']};--ink:{p['ink']};--muted:{p['muted']};--accent:{p['accent']};--bad:{p['bad']}">
      <svg viewBox="0 22 400 126" role="img" aria-label="Silueta {kind}">
        <line class="road" x1="0" y1="141" x2="400" y2="141"/>
        <g class="bob">
          <path d="{car['body']}" fill="var(--ink)"/>
          {windows}
          <rect x="{car['front'] - 9}" y="{car['head_y'] - 3}" width="9" height="6" rx="2" fill="var(--accent)"/>
          <rect x="{car['rear']}" y="{car['tail_y'] - 3}" width="6" height="8" rx="2" fill="var(--bad)"/>
          {wheels}
        </g>
      </svg>
    </div>"""
