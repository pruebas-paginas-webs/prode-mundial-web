# Prode Mundial 2026

Página estática (GitHub Pages) con predicciones de la Ronda de 32 del Mundial 2026:
**https://pruebas-paginas-webs.github.io/prode-mundial-web/**

A diferencia de la versión original (HTML con números hardcodeados y sin pipeline),
ahora el modelo vive en el repo y la página se regenera desde los datos.

## Archivos

| Archivo      | Qué es |
|--------------|--------|
| `data.json`  | Entradas: cuotas 1X2 reales por partido (con casa y fecha), ratings Elo/FIFA por equipo (con nota de fuente) y parámetros del modelo. |
| `model.py`   | El modelo (solo librería estándar de Python). Lee `data.json`, escribe `index.html` e imprime una tabla de verificación. |
| `index.html` | Salida generada. No editar a mano. |

## Método (por partido)

1. **De-vig**: las cuotas 1X2 se convierten a probabilidades y se les quita el
   margen de la casa con el **método de potencias** (mejor que la normalización
   proporcional frente al sesgo favorito-longshot).
2. **Ajuste al mercado**: se buscan las λ (goles esperados) de un **Poisson
   bivariado con corrección Dixon-Coles** (ρ = −0.10) cuya matriz de marcadores
   reproduce esas probabilidades. De ahí salen el **xG implícito** y el
   **marcador más probable** (condicionado al resultado más probable).
3. **Pata de ratings**: independiente del mercado, con ratings Elo/FIFA
   (escala SUM, `dr/600`) y **bonus de anfitrión** (+80 Elo para México en el
   Azteca y EE.UU. en Levi's). Razón de fuerzas `10^(dr/600)` repartiendo un
   total de 2.55 goles.
4. **Mezcla**: `p_final = 0.72 · mercado + 0.28 · modelo`. El peso alto del
   mercado es deliberado: la evidencia empírica es que las cuotas agregan más
   información que cualquier modelo público de ratings.
5. **Value**: `edge = p_final × cuota − 1`; se publica si supera el 3%, con
   stake de **Kelly ¼** sobre bankroll de 100u (tope 20u).
6. **Pasa de ronda**: `p_gana + 0.5 × p_empate` (alargue/penales ≈ moneda).

## Actualizar la página

1. Editar `data.json`: refrescar cuotas (son un snapshot y se mueven), sumar
   partidos de la siguiente ronda o corregir ratings.
2. Regenerar y revisar la tabla de verificación:

   ```bash
   python3 model.py
   ```

3. Commit y push a `main` (GitHub Pages publica solo).

## Limitaciones honestas

- Las cuotas provienen de coberturas de prensa (FanDuel/DraftKings, 27–28/06),
  no de una API en vivo; pueden diferir levemente del precio actual.
- Los ratings son la última tabla completa disponible, reordenada con las
  posiciones publicadas en junio 2026 (detalle en `ratings_note` de `data.json`).
- El "value" señala divergencia modelo↔mercado, no dinero gratis. **No es
  consejo de apuestas.**
