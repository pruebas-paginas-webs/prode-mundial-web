#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prode Mundial 2026 — pipeline del modelo.

Lee data.json (cuotas 1X2 reales + ratings Elo/FIFA) y regenera index.html.

Método, por partido:
  1. De-vig de las cuotas por el método de potencias (corrige el sesgo
     favorito-longshot mejor que la normalización proporcional).
  2. Ajuste de un Poisson bivariado con corrección Dixon-Coles a las
     probabilidades del mercado -> lambdas (xG implícito) y matriz de
     marcadores. El pick es el marcador de máxima probabilidad.
  3. Pata de modelo independiente: ratings Elo/FIFA (escala dr/600, bonus
     de anfitrión) -> lambdas -> probabilidades Dixon-Coles.
  4. Mezcla: p_final = w*mercado + (1-w)*modelo  (w en data.json).
  5. Value: edge = p_final*cuota - 1; se publica si supera el umbral, con
     stake de Kelly fraccionado (1/4) sobre bankroll de 100u, tope 20u.
  6. "Pasa de ronda" = p_gana + 0.5*p_empate (alargue/penales ~ moneda).

Uso:  python3 model.py          (reescribe index.html e imprime verificación)
Solo librería estándar; sin dependencias.
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- utilidades

def poisson_vec(lam, maxg):
    return [math.exp(-lam) * lam ** k / math.factorial(k) for k in range(maxg + 1)]


def dc_probs(lh, la, rho, maxg, want_matrix=False):
    """Probabilidades 1/X/2 (y matriz opcional) de un Poisson con tau Dixon-Coles."""
    ph = poisson_vec(lh, maxg)
    pa = poisson_vec(la, maxg)
    M = [[ph[i] * pa[j] for j in range(maxg + 1)] for i in range(maxg + 1)]
    M[0][0] *= max(0.0, 1.0 - lh * la * rho)
    M[0][1] *= 1.0 + lh * rho
    M[1][0] *= 1.0 + la * rho
    M[1][1] *= 1.0 - rho
    tot = sum(sum(row) for row in M)
    pw = px = pl = 0.0
    for i in range(maxg + 1):
        for j in range(maxg + 1):
            M[i][j] /= tot
            if i > j:
                pw += M[i][j]
            elif i == j:
                px += M[i][j]
            else:
                pl += M[i][j]
    if want_matrix:
        return (pw, px, pl), M
    return (pw, px, pl)


def devig_power(odds):
    """Cuotas decimales -> probabilidades sin margen (metodo de potencias)."""
    raw = [1.0 / o for o in odds]
    lo, hi = 1.0, 6.0
    for _ in range(60):
        k = (lo + hi) / 2.0
        s = sum(p ** k for p in raw)
        if s > 1.0:
            lo = k
        else:
            hi = k
    k = (lo + hi) / 2.0
    q = [p ** k for p in raw]
    s = sum(q)
    return [p / s for p in q]


def fit_lambdas(target, rho, maxg):
    """Busca (lh, la) cuya matriz DC reproduce las prob. de mercado (1X2)."""
    def sse(lh, la):
        pw, px, pl = dc_probs(lh, la, rho, maxg)
        return (pw - target[0]) ** 2 + (px - target[1]) ** 2 + (pl - target[2]) ** 2

    best = (1.3, 1.3)
    best_err = sse(*best)
    for step, span in ((0.08, None), (0.02, 0.09), (0.004, 0.022)):
        if span is None:
            grid_h = [0.10 + i * step for i in range(int(3.7 / step) + 1)]
            grid_a = grid_h
        else:
            bh, ba = best
            grid_h = [max(0.02, bh + (i - int(span / step)) * step)
                      for i in range(2 * int(span / step) + 1)]
            grid_a = [max(0.02, ba + (i - int(span / step)) * step)
                      for i in range(2 * int(span / step) + 1)]
        for lh in grid_h:
            for la in grid_a:
                e = sse(lh, la)
                if e < best_err:
                    best_err, best = e, (lh, la)
    return best[0], best[1], math.sqrt(best_err / 3.0)


def rating_lambdas(r_home, r_away, host, prm):
    """Ratings -> lambdas: razon de fuerzas 10^(dr/600), total de goles fijo."""
    dr = r_home - r_away
    if host == "home":
        dr += prm["host_elo_bonus"]
    elif host == "away":
        dr -= prm["host_elo_bonus"]
    ratio = 10.0 ** (dr / prm["elo_scale"])
    lh = prm["total_goals"] * ratio / (1.0 + ratio)
    return lh, prm["total_goals"] - lh


def pct_triple(p):
    """Redondea a enteros que suman 100 (ajusta en el mayor)."""
    r = [round(x * 100) for x in p]
    r[r.index(max(r))] += 100 - sum(r)
    return r


# ------------------------------------------------------------------- calculo

def compute(data):
    prm = data["params"]
    rows = []
    for m in data["matches"]:
        odds = [m["odds"]["h"], m["odds"]["d"], m["odds"]["a"]]
        market = devig_power(odds)
        lh_mkt, la_mkt, fit_err = fit_lambdas(market, prm["dc_rho"], prm["max_goals"])
        _, matrix = dc_probs(lh_mkt, la_mkt, prm["dc_rho"], prm["max_goals"], True)

        lh_elo, la_elo = rating_lambdas(
            data["ratings"][m["home"]], data["ratings"][m["away"]], m["host"], prm)
        model = list(dc_probs(lh_elo, la_elo, prm["dc_rho"], prm["max_goals"]))

        w = prm["market_weight"]
        blend = [w * mk + (1 - w) * md for mk, md in zip(market, model)]

        # Pick: resultado más probable según la mezcla, y el marcador más
        # probable condicionado a ese resultado (el modal global de un DC
        # suele ser 1-1 aun con favorito claro, confuso para un prode).
        outcome = blend.index(max(blend))  # 0=local, 1=empate, 2=visitante
        best_i, best_j, best_p = 0, 0, -1.0
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                ok = (i > j, i == j, i < j)[outcome]
                if ok and matrix[i][j] > best_p:
                    best_i, best_j, best_p = i, j, matrix[i][j]

        value = []
        labels = [m["home"], "Empate", m["away"]]
        for p, o, lab in zip(blend, odds, labels):
            edge = p * o - 1.0
            if edge >= prm["value_threshold"]:
                stake = prm["kelly_fraction"] * edge / (o - 1.0) * 100.0
                stake = min(prm["max_stake_u"], max(0.5, round(stake * 2) / 2))
                value.append({"label": lab, "odds": o, "edge": edge, "stake": stake})

        rows.append({
            "m": m, "market": market, "model": model, "blend": blend,
            "lh": lh_mkt, "la": la_mkt, "fit_err": fit_err,
            "pick": (best_i, best_j), "pick_p": best_p, "value": value,
            "adv": blend[0] + 0.5 * blend[1],
        })
    return rows


# -------------------------------------------------------------------- render

CSS = """
  :root{
    --bg:#070b15; --bg2:#0c1322; --card:#101a2e; --line:rgba(255,255,255,.09);
    --ink:#eef3fc; --muted:#9aa7c2;
    --home:#34d399; --draw:#9aa7c2; --away:#8f97f3; --accent:#fbbf24; --value:#34d399;
    --display:'Space Grotesk',system-ui,sans-serif; --body:'Inter',system-ui,sans-serif;
  }
  *{box-sizing:border-box;}
  body{
    margin:0; background:var(--bg); color:var(--ink); font-family:var(--body); font-size:16px;
    -webkit-font-smoothing:antialiased; line-height:1.5; font-variant-numeric:tabular-nums;
    background-image:radial-gradient(900px 380px at 50% -120px, rgba(99,102,241,.18), transparent 70%);
  }
  .wrap{max-width:640px; margin:0 auto; padding:max(26px,env(safe-area-inset-top)) 16px calc(48px + env(safe-area-inset-bottom));}
  header{margin:6px 4px 22px;}
  .kicker{font:600 12px/1 var(--body); letter-spacing:.18em; text-transform:uppercase; color:var(--accent);}
  h1{font-family:var(--display); font-weight:700; font-size:30px; letter-spacing:-.02em; margin:8px 0 4px;}
  .sub{color:var(--muted); font-size:13px; margin:0;}
  .note{margin:14px 4px 0; padding:10px 12px; font-size:12px; color:#fde9b8; line-height:1.5;
        background:rgba(251,191,36,.08); border:1px solid rgba(251,191,36,.28); border-radius:12px;}
  .grid{display:grid; gap:14px; grid-template-columns:1fr;}
  @media(min-width:560px){ .grid{grid-template-columns:1fr 1fr;} .wrap{max-width:880px;} }
  .card{
    background:linear-gradient(180deg,var(--card),var(--bg2)); border:1px solid var(--line);
    border-radius:18px; padding:16px 16px 15px; box-shadow:0 8px 28px rgba(0,0,0,.35);
    animation:rise .42s ease both; animation-delay:calc(var(--i) * 35ms);
  }
  @keyframes rise{from{opacity:0; transform:translateY(10px);} to{opacity:1; transform:none;}}
  @media(prefers-reduced-motion:reduce){ .card{animation:none;} }
  .card-top{display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; min-height:20px;}
  .date{font:600 12px/1 var(--body); letter-spacing:.05em; color:var(--muted);}
  .flag{font:600 11px/1 var(--body); letter-spacing:.07em; text-transform:uppercase; color:var(--accent);
        background:rgba(251,191,36,.12); border:1px solid rgba(251,191,36,.32); padding:4px 9px; border-radius:999px;}
  .teams{display:flex; align-items:baseline; gap:8px; font-family:var(--display); margin:0 0 6px; font-weight:600;}
  .team{font-size:18px; letter-spacing:-.01em;}
  .vs{color:var(--muted); font-size:12px; font-weight:500;}
  .venue{display:flex; align-items:center; gap:5px; font-size:12px; color:var(--muted); margin:0 0 12px;}
  .bar{display:flex; gap:3px; height:10px; margin-bottom:10px;}
  .seg{border-radius:3px; min-width:6px;}
  .seg.home{background:var(--home);} .seg.draw{background:var(--draw);} .seg.away{background:var(--away);}
  .legend{display:flex; flex-wrap:wrap; gap:6px 14px; font-size:12.5px; color:var(--muted); margin-bottom:10px;}
  .lg strong{color:var(--ink); font-weight:600;}
  .dot{display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; vertical-align:middle;}
  .dot.home{background:var(--home);} .dot.draw{background:var(--draw);} .dot.away{background:var(--away);}
  .sources{font-size:12px; color:var(--muted); margin:0 0 12px; letter-spacing:.01em;}
  .adv{margin-top:-8px;}
  .pickrow{display:flex; justify-content:space-between; align-items:center; gap:10px;
           padding-top:12px; border-top:1px solid var(--line);}
  .pick{font-family:var(--display); font-weight:600; font-size:15px;}
  .pick-score{display:inline-block; margin-left:6px; font-family:var(--body); font-weight:700; font-size:12px; padding:2px 7px; border-radius:6px;}
  .ps-home{background:var(--home); color:#04130c;}
  .ps-away{background:var(--away); color:#0a0a2e;}
  .ps-draw{background:var(--draw); color:#0b1220;}
  .xg{font-size:12.5px; color:var(--muted);}
  .value{display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin-top:12px;
         background:rgba(52,211,153,.09); border:1px solid rgba(52,211,153,.28); border-radius:12px; padding:10px;}
  .value-tag{font:700 10px/1 var(--body); letter-spacing:.1em; color:#04130c; background:var(--value); padding:4px 7px; border-radius:6px;}
  .value-bet{font-size:12.5px; color:#bdf3da;}
  .value-bet b{color:#ecfdf5;}
  footer{margin:28px 6px 0; color:var(--muted); font-size:12px; text-align:center; line-height:1.6;}
"""


def render(data, rows):
    prm = data["params"]
    w = int(round(prm["market_weight"] * 100))
    cards = []
    for idx, r in enumerate(rows):
        m = r["m"]
        b = pct_triple(r["blend"])
        mk = pct_triple(r["market"])
        md = pct_triple(r["model"])
        i, j = r["pick"]
        if i > j:
            pick_lab, pick_cls = m["home"], "ps-home"
        elif i < j:
            pick_lab, pick_cls = m["away"], "ps-away"
        else:
            pick_lab, pick_cls = "Empate", "ps-draw"

        if m["host"]:
            chip = '<span class="flag">anfitrión</span>'
        elif max(r["blend"]) < 0.5:
            chip = '<span class="flag">parejo</span>'
        else:
            chip = "<span></span>"

        if r["value"]:
            bets = "".join(
                '<span class="value-bet"><b>{}</b> @{:.2f} · edge +{:.1f}% · {:g}u</span>'.format(
                    v["label"], v["odds"], v["edge"] * 100, v["stake"])
                for v in r["value"])
            value_html = ('\n    <div class="value">\n      <span class="value-tag">VALUE</span>\n'
                          "      {}\n    </div>".format(bets))
        else:
            value_html = ""

        cards.append("""
  <article class="card" style="--i:{i}">
    <div class="card-top">
      <span class="date">{date}</span>
      {chip}
    </div>
    <h2 class="teams"><span class="team">{home}</span> <span class="vs">vs</span> <span class="team">{away}</span></h2>
    <p class="venue">📍 {venue}</p>
    <div class="bar" role="img" aria-label="{home} {b0} por ciento, empate {b1} por ciento, {away} {b2} por ciento">
      <span class="seg home" style="flex:{f0:.4f}"></span>
      <span class="seg draw" style="flex:{f1:.4f}"></span>
      <span class="seg away" style="flex:{f2:.4f}"></span>
    </div>
    <div class="legend">
      <span class="lg"><b class="dot home"></b>{home} <strong>{b0}%</strong></span>
      <span class="lg"><b class="dot draw"></b>Empate <strong>{b1}%</strong></span>
      <span class="lg"><b class="dot away"></b>{away} <strong>{b2}%</strong></span>
    </div>
    <p class="sources">mercado ({book}) {mk0}/{mk1}/{mk2} · modelo Elo {md0}/{md1}/{md2} · mezcla {w}/{wc}</p>
    <p class="sources adv">Pasa de ronda: {home} <b>{adv:.0f}%</b> · {away} <b>{advc:.0f}%</b></p>
    <div class="pickrow">
      <span class="pick">{pick_lab} <span class="pick-score {pick_cls}">{pi}-{pj}</span></span>
      <span class="xg">xG impl. {lh:.2f}–{la:.2f}</span>
    </div>{value}
  </article>""".format(
            i=idx, date=m["date"], chip=chip, home=m["home"], away=m["away"],
            venue=m["venue"], b0=b[0], b1=b[1], b2=b[2],
            f0=r["blend"][0], f1=r["blend"][1], f2=r["blend"][2],
            book=m["book"], mk0=mk[0], mk1=mk[1], mk2=mk[2],
            md0=md[0], md1=md[1], md2=md[2], w=w, wc=100 - w,
            adv=r["adv"] * 100, advc=(1 - r["adv"]) * 100,
            pick_lab=pick_lab, pick_cls=pick_cls, pi=i, pj=j,
            lh=r["lh"], la=r["la"], value=value_html))

    return """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#070b15">
<title>Prode Mundial 2026 · Ronda de 32</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">
<style>{css}</style>
</head>
<body>
  <div class="wrap">
    <header>
      <div class="kicker">⚽ Predicciones</div>
      <h1>Prode Mundial 2026</h1>
      <p class="sub">Ronda de 32 · Poisson/Dixon-Coles calibrado al mercado + ratings Elo · mezcla {w}% mercado / {wc}% modelo · Generado {gen}</p>
      <p class="note"><b>Método:</b> las cuotas 1X2 reales se limpian de margen (método de potencias) y se ajusta
      un Poisson con corrección Dixon-Coles que las reproduce → de ahí salen el <b>xG implícito</b> y el
      <b>marcador más probable</b>. Una pata independiente de <b>ratings Elo/FIFA</b> (con bonus de anfitrión)
      se mezcla {w}/{wc} con el mercado para las barras. <b>VALUE</b> = la mezcla le asigna a un resultado más
      probabilidad que la implícita en el precio (stake = ¼ Kelly, tope 20u). Pipeline completo en
      <code>model.py</code> + <code>data.json</code> del repo. No es consejo de apuestas.</p>
    </header>
    <div class="grid">
{cards}
    </div>
    <footer>
      Cuotas: FanDuel/DraftKings 27–28/06 (snapshot de prensa; se mueven). Ratings: tabla FIFA/Elo documentada en data.json.
      Regenerar: <code>python3 model.py</code>. Probabilidades, no certezas — no es consejo de apuestas.
    </footer>
  </div>
</body>
</html>
""".format(css=CSS, w=w, wc=100 - w, gen=data["generated"], cards="".join(cards))


# ---------------------------------------------------------------------- main

def main():
    with open(os.path.join(HERE, "data.json"), encoding="utf-8") as fh:
        data = json.load(fh)
    rows = compute(data)

    print("=== Verificación del ajuste (todas las prob. en %) ===")
    hdr = "{:<28} {:>13} {:>13} {:>13} {:>11} {:>7} {}"
    print(hdr.format("partido", "mercado", "modelo", "mezcla", "xG(fit)", "errfit", "value"))
    for r in rows:
        m = r["m"]
        fmt = lambda p: "/".join(str(x) for x in pct_triple(p))
        vals = ", ".join("{} +{:.1f}%".format(v["label"], v["edge"] * 100) for v in r["value"]) or "-"
        print(hdr.format(
            "{} vs {}".format(m["home"][:12], m["away"][:12]),
            fmt(r["market"]), fmt(r["model"]), fmt(r["blend"]),
            "{:.2f}-{:.2f}".format(r["lh"], r["la"]),
            "{:.4f}".format(r["fit_err"]), vals))

    out = render(data, rows)
    with open(os.path.join(HERE, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(out)
    print("\nindex.html regenerado ({} tarjetas).".format(len(rows)))


if __name__ == "__main__":
    main()
