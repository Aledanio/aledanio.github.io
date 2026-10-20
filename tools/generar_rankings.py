#!/usr/bin/env python3
"""Genera el vocabulario y los rankings diarios de Aledaño.

Uso (desde la raíz del repositorio):
    python3 -m venv .venv
    .venv/bin/pip install "spacy>=3.7,<3.8" wordfreq
    .venv/bin/python -m spacy download es_core_news_lg
    .venv/bin/python tools/generar_rankings.py

Para vetar palabras, añádelas a tools/excluir.txt (una por línea) y vuelve a
ejecutar el script. Para añadir días, amplía la lista DIAS.
"""

import json
import re
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import spacy
from wordfreq import top_n_list

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "data"
EXCLUIR = Path(__file__).resolve().parent / "excluir.txt"

INICIO = date(2026, 10, 21)
DIAS = [
    "castaña", "biblioteca", "tren", "abuela", "guitarra",
    "nieve", "espejo", "queso", "reloj", "calabaza",
]

N_FRECUENTES = 60000
SOLO_LETRAS = re.compile(r"^[a-záéíóúüñ]{3,}$")
CATEGORIAS = {"NOUN", "VERB", "ADJ", "ADV"}

# Palabrotas e insultos que no queremos en el juego. tools/excluir.txt amplía esta lista.
PALABROTAS = {
    "puta", "puto", "putas", "putos", "putada", "puticlub", "putero", "mierda",
    "coño", "joder", "jodido", "jodida", "cojones", "cojón", "polla", "pollas",
    "gilipollas", "gilipollez", "cabrón", "cabrona", "cabrones", "hostia",
    "hostias", "follar", "follando", "follada", "zorra", "zorras", "maricón",
    "marica", "mariconazo", "bollera", "hijoputa", "capullo", "cagar", "cagada",
    "cagón", "mamón", "mamada", "pajillero", "cipote", "chocho",
    "verga", "pendejo", "pendeja", "culero", "chingar", "chingada",
    "chingado", "huevón", "boludo", "pelotudo", "carajo", "putita",
    "culo", "picha", "orto", "ojete",
    "correrse", "subnormal",
    "mongólico", "sudaca", "negrata", "panchito", "facha", "zorrón",
    "putón", "guarra", "guarro", "furcia", "ramera",
}


def leer_excluidas():
    if not EXCLUIR.exists():
        return set()
    lineas = EXCLUIR.read_text(encoding="utf-8").splitlines()
    return {l.strip().lower() for l in lineas if l.strip() and not l.startswith("#")}


def construir_vocabulario(nlp, excluidas):
    candidatas = [
        w for w in top_n_list("es", N_FRECUENTES)
        if SOLO_LETRAS.match(w) and w not in excluidas and w not in PALABROTAS
    ]
    candidatas = [w for w in candidatas if nlp.vocab.has_vector(w)]
    print(f"Candidatas con vector: {len(candidatas)}", file=sys.stderr)

    # Analiza cada palabra suelta: categoría y lema.
    analisis = {}
    for doc in nlp.pipe(candidatas, batch_size=2000):
        if len(doc) != 1:
            continue
        tok = doc[0]
        analisis[tok.text] = (tok.pos_, tok.lemma_.lower(), tok.is_stop)

    # Sin contexto, el etiquetador marca muchos sustantivos comunes como PROPN, así que
    # los nombres propios se detectan aparte, con es_nombre_propio().
    validas = [
        w for w in candidatas
        if w in analisis
        and analisis[w][0] in CATEGORIAS | {"PROPN"}
        and not analisis[w][2]
        and not es_nombre_propio(nlp, w)
    ]
    conjunto = set(validas)

    # Una forma por lema: el propio lema si está entre las válidas; si no, la más frecuente.
    por_lema = {}
    for w in validas:  # ya van de más a menos frecuente
        lema = lema_de(w, analisis[w], conjunto, analisis)
        if lema in excluidas or lema in PALABROTAS:
            continue
        if lema not in por_lema:
            por_lema[lema] = lema if lema in conjunto else w

    # El juego busca sin tildes: si dos formas coinciden al quitarlas (librería/libreria),
    # se queda la más frecuente, salvo que la otra sea palabra de algún día.
    por_clave = {sin_tildes(p): p for p in DIAS}
    for w in por_lema.values():
        por_clave.setdefault(sin_tildes(w), w)
    elegidas = set(por_clave.values())
    vocab = [w for w in dict.fromkeys(por_lema.values()) if w in elegidas]
    for palabra in DIAS:
        if palabra not in vocab:
            print(f"Añadiendo palabra del día que faltaba: {palabra}", file=sys.stderr)
            vocab.append(palabra)
    return vocab


def es_nombre_propio(nlp, w):
    """Madrid, Juan, Google…: la forma con mayúscula es mucho más frecuente que en minúscula.

    La tabla de vectores está ordenada por frecuencia, así que la fila sirve de rango.
    """
    filas = nlp.vocab.vectors.key2row
    minuscula = filas.get(nlp.vocab.strings[w], -1)
    mayuscula = filas.get(nlp.vocab.strings.add(w.capitalize()), -1)
    return mayuscula >= 0 and mayuscula * 3 < minuscula


def lema_de(w, datos, conjunto, analisis):
    """Lema de spaCy, corregido para plurales y femeninos de adjetivos que falla sin contexto."""
    pos, lema, _ = datos
    if w.endswith("es") and w[:-2] in conjunto:
        return analisis[w[:-2]][1]
    if w.endswith("s") and w[:-1] in conjunto:
        return analisis[w[:-1]][1]
    if pos == "ADJ":
        for fem, masc in (("as", "os"), ("a", "o")):
            if w.endswith(fem) and w[: -len(fem)] + masc in conjunto:
                return analisis[w[: -len(fem)] + masc][1]
    return lema


def sin_tildes(w):
    return w.translate(str.maketrans("áéíóúü", "aeiouu"))


def main():
    nlp = spacy.load("es_core_news_lg", exclude=["parser", "ner"])
    excluidas = leer_excluidas() - set(DIAS)
    vocab = construir_vocabulario(nlp, excluidas)
    print(f"Vocabulario final: {len(vocab)} palabras", file=sys.stderr)

    matriz = np.array([nlp.vocab.get_vector(w) for w in vocab], dtype=np.float32)
    matriz /= np.linalg.norm(matriz, axis=1, keepdims=True)
    indice = {w: i for i, w in enumerate(vocab)}

    DATOS.mkdir(exist_ok=True)
    (DATOS / "vocabulario.json").write_text(
        json.dumps(vocab, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )

    for num, palabra in enumerate(DIAS, start=1):
        i = indice[palabra]
        similitud = matriz @ matriz[i]
        similitud[i] = 2.0  # la secreta, siempre la primera
        orden = np.argsort(-similitud, kind="stable").tolist()
        dia = {
            "num": num,
            "fecha": (INICIO + timedelta(days=num - 1)).isoformat(),
            "palabra": palabra,
            "orden": orden,
        }
        (DATOS / f"dia-{num:03d}.json").write_text(
            json.dumps(dia, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
        )
        cercanas = ", ".join(vocab[j] for j in orden[1:41])
        print(f"\n#{num} {dia['fecha']} · {palabra}\n{cercanas}")


if __name__ == "__main__":
    main()
