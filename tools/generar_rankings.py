#!/usr/bin/env python3
"""Genera el vocabulario y los rankings diarios de Aledaño.

Uso (desde la raíz del repositorio):
    python3 -m venv .venv
    .venv/bin/pip install "spacy>=3.7,<3.8" wordfreq spylls
    .venv/bin/python -m spacy download es_core_news_lg
    .venv/bin/python tools/generar_rankings.py

Para vetar palabras, añádelas a tools/excluir.txt (una por línea) y vuelve a
ejecutar el script. Para rescatar palabras comunes que el script toma por
nombres propios (el Ayuntamiento), añádelas a tools/incluir.txt. Los plurales
cuyo singular está en el vocabulario salen de él, salvo los de significado
propio (gafas, martes), que van en tools/plurales_propios.txt. Para añadir
días, amplía la lista DIAS.

Solo entran palabras que reconoce el diccionario ortográfico de español de
LibreOffice (proyecto RLA-ES, que sigue la norma de la RAE), guardado en
tools/diccionario/ con su licencia. Así quedan fuera extranjerismos y erratas.
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
from spylls.hunspell import Dictionary
from wordfreq import top_n_list, zipf_frequency

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "data"
EXCLUIR = Path(__file__).resolve().parent / "excluir.txt"
INCLUIR = Path(__file__).resolve().parent / "incluir.txt"
PLURALES_PROPIOS = Path(__file__).resolve().parent / "plurales_propios.txt"
DICCIONARIO = Path(__file__).resolve().parent / "diccionario" / "es"

INICIO = date(2026, 10, 11)
DIAS = [
    "castaña", "biblioteca", "tren", "abuela", "guitarra",
    "nieve", "espejo", "queso", "reloj", "calabaza",
    "faro", "mariposa", "café", "montaña", "sombrero",
    "perro", "cuchillo", "maleta", "bosque", "volcán",
    # Del 31 de octubre al 31 de diciembre de 2026 (nº 21 a 82)
    "fantasma", "cementerio", "lápiz", "ballena", "cocina",
    "luna", "castillo", "zapato", "tomate", "pirata",
    "tormenta", "teléfono", "caballo", "miel", "isla",
    "escalera", "médico", "seta", "película", "búho",
    "llave", "sopa", "planeta", "iglesia", "pintor",
    "almohada", "tiburón", "periódico", "limón", "rey",
    "puente", "bufanda", "pájaro", "ordenador", "tortilla",
    "brújula", "teatro", "oveja", "lámpara", "río",
    "huevo", "circo", "martillo", "desierto", "carta",
    "jabón", "granja", "cohete", "chocolate", "corona",
    "chimenea", "reno", "lotería", "turrón", "villancico",
    "regalo", "campana", "manta", "broma", "tesoro",
    "dragón", "uva",
]

N_FRECUENTES = 150000
PENALIZACION_RARAS = 0.05
SOLO_LETRAS = re.compile(r"^[a-záéíóúüñ]{3,}$")
# Risas escritas (jaja, jajajaja, jejeje…): no son palabras del juego.
RISA = re.compile(r"^(?:(?:ja){2,}|(?:je){2,}|(?:ji){2,}|(?:jo){2,})j?$")
CATEGORIAS = {"NOUN", "VERB", "ADJ", "ADV"}
# Palabras vacías de spaCy que sí tienen significado propio y deben estar en el juego.
CONSERVAR = {
    "acuerdo", "cuenta", "día", "estado", "fin", "final", "lado", "manera", "medio", "modo",
    "parte", "uso", "verdad", "vez", "tarde", "bueno", "grande", "nuevo", "largo", "breve", "claro",
    "igual", "total", "propio", "posible", "diferente", "verdadero", "anterior", "siguiente",
    "último", "mayor", "mejor", "peor", "conocer", "conseguir", "dar", "decir", "hacer",
    "llevar", "poner", "realizar", "saber", "tener", "usar", "ver",
}
INFINITIVO = re.compile(r"(?:ar|er|ir|ír)$")
# Terminaciones de palabras derivadas de un verbo que son sustantivos o adjetivos.
DERIVADA = re.compile(
    r"(?:ción|sión|miento|ble|dor|dora|ancia|encia|ura|aje|ista|ismo|dad|ivo|iva|orio|oria)e?s?$"
)

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


def leer_lista(ruta):
    if not ruta.exists():
        return set()
    lineas = ruta.read_text(encoding="utf-8").splitlines()
    return {l.strip().lower() for l in lineas if l.strip() and not l.startswith("#")}


# Palabras que se usan en plural y no deben fundirse con su singular (martes ≠ marte).
PLURALES = leer_lista(PLURALES_PROPIOS)
# Plurales que cambian la tilde de sitio.
PLURALES_IRREGULARES = {"régimen": "regímenes", "carácter": "caracteres", "espécimen": "especímenes"}


def construir_vocabulario(nlp, excluidas, incluidas):
    diccionario = Dictionary.from_files(str(DICCIONARIO))
    candidatas = [
        w for w in top_n_list("es", N_FRECUENTES)
        if SOLO_LETRAS.match(w) and not RISA.match(w) and w not in excluidas and w not in PALABROTAS
    ]
    # Con una palabra se veta también su plural (hable y hables, ble y bles).
    candidatas = [
        w for w in candidatas
        if w in PLURALES or not (w.endswith("s") and (w[:-1] in excluidas or w.endswith("es") and w[:-2] in excluidas))
    ]
    candidatas += [w for w in incluidas if w not in set(candidatas)]
    candidatas = [w for w in candidatas if nlp.vocab.has_vector(w)]
    print(f"Candidatas con vector: {len(candidatas)}", file=sys.stderr)
    # El diccionario deja fuera extranjerismos (trekking), erratas (libreria) y algunos nombres propios.
    candidatas = [w for w in candidatas if w in incluidas or diccionario.lookup(w)]
    print(f"Candidatas en el diccionario: {len(candidatas)}", file=sys.stderr)
    # Formas sin tilde que el diccionario acepta (arboles, lamparas): si la misma palabra
    # con tilde es más frecuente, se queda solo esa. Si la forma sin tilde es la más
    # frecuente, conviven las dos (papa y papá). La ñ no se toca: campana y campaña.
    primera = {}
    for w in candidatas:  # de más a menos frecuente
        primera.setdefault(sin_tildes(w), w)
    candidatas = [w for w in candidatas if primera[sin_tildes(w)] == w or w != sin_tildes(w) or w in incluidas]

    # Analiza cada palabra suelta: categoría y lema.
    analisis = {}
    for doc in nlp.pipe(candidatas, batch_size=2000):
        if len(doc) != 1:
            continue
        tok = doc[0]
        analisis[tok.text] = (tok.pos_, tok.lemma_.lower(), tok.is_stop)

    # Las que spaCy toma por verbo sueltas se miran también tras un artículo: «la pregunta»
    # es un sustantivo y «la iba» no. nominal[w] cuenta en cuántos de los cuatro marcos
    # sale como sustantivo o adjetivo.
    dudosas = [w for w in candidatas if w in analisis and analisis[w][0] in ("VERB", "AUX", "PROPN")]
    marcos = [f"{articulo} {w}" for w in dudosas for articulo in ("el", "la", "un", "una")]
    nominal = dict.fromkeys(dudosas, 0)
    for frase, doc in zip(marcos, nlp.pipe(marcos, batch_size=4000)):
        if doc[-1].pos_ in ("NOUN", "ADJ", "PROPN"):
            nominal[frase.split(" ", 1)[1]] += 1

    # Sin contexto, el etiquetador marca muchos sustantivos comunes como PROPN, así que
    # los nombres propios se detectan aparte, con es_nombre_propio().
    validas, propios = [], []
    for w in candidatas:
        if w not in analisis:
            continue
        if w not in incluidas:
            pos, _, vacia = analisis[w]
            if pos not in CATEGORIAS | {"PROPN"} or (vacia and w not in CONSERVAR):
                continue
            if es_forma_verbal(diccionario, w, analisis[w], nominal.get(w, 4)):
                continue
            if es_nombre_propio(nlp, w):
                propios.append(w)
                continue
        validas.append(w)
    conjunto = set(validas)
    print("Descartadas como nombre propio (las 150 más frecuentes; las que sean comunes van a "
          "tools/incluir.txt):\n  " + ", ".join(propios[:150]), file=sys.stderr)

    # Una forma por lema: el propio lema si está entre las válidas; si no, la más frecuente.
    por_lema = {}
    for w in validas:  # ya van de más a menos frecuente
        lema = w if w in incluidas else lema_de(w, analisis, conjunto)
        if lema in excluidas or lema in PALABROTAS:
            continue
        if lema not in por_lema:
            por_lema[lema] = lema if lema in conjunto else w

    vocab = list(dict.fromkeys(por_lema.values()))
    for palabra in DIAS:
        if palabra not in vocab:
            print(f"Añadiendo palabra del día que faltaba: {palabra}", file=sys.stderr)
            vocab.append(palabra)

    # Los plurales cuentan como su singular: fuera los que aún conviven con él (dioses y dios).
    conjunto = set(vocab)
    por_clave = {}
    for w in vocab:
        por_clave.setdefault(sin_tildes(w), []).append(w)
    fundidos = {}
    for w in vocab:
        singular = None if w in PLURALES else singular_de(w, conjunto, por_clave)
        if singular:
            fundidos[w] = singular
    print(f"Plurales que salen del vocabulario ({len(fundidos)}):\n  "
          + ", ".join(f"{p} → {s}" for p, s in fundidos.items()), file=sys.stderr)
    return [w for w in vocab if w not in fundidos], fundidos


def singular_de(w, conjunto, por_clave):
    """Singular de w si es un plural (-s, -es, -ces) y el singular está en el vocabulario.

    La tilde solo puede cambiar como en los plurales de verdad: la pierde la última sílaba
    (camión → camiones), la gana la antepenúltima (joven → jóvenes) o se mueve en unos
    pocos irregulares (régimen → regímenes). Así inglés no pasa por plural de ingle, ni
    mamas de mamá.
    """
    if w.endswith("ces") and w[:-3] + "z" in conjunto:
        return w[:-3] + "z"  # luces → luz
    for corte in (2, 1):  # árboles → árbol, casas → casa
        if w.endswith("es"[-corte:]) and w[:-corte] in conjunto:
            return w[:-corte]
    if w.endswith("es"):
        for c in por_clave.get(sin_tildes(w[:-2]), []):
            if PLURALES_IRREGULARES.get(c) == w:
                return c
            if not re.search("[áéíóú]", c) and len(re.findall("[áéíóú]", w)) == 1:
                return c  # jóvenes, órdenes
            if re.search("[áéíóú][ns]$", c) and not re.search("[áéíóú]", w):
                return c  # marrones
    return None


def es_forma_verbal(diccionario, w, datos, nominal):
    """Cantaba, iba, quemaste, confesarme, déjale…: de los verbos solo entra el infinitivo.

    Solo se mira lo que spaCy toma por verbo (o por nombre propio, como nieva). Es forma
    verbal si no es un infinitivo y, o bien todas sus lecturas en el diccionario salen de
    un infinitivo distinto, o bien tras un artículo no suena a sustantivo (nominal < 2).
    Pregunta, charla o parada se quedan; administración o cantante salen de un verbo,
    pero son palabras derivadas, y también se quedan. Por último, si su infinitivo se usa
    más que ella (das/dar, jugando/jugar), es una conjugación; pregunta o denuncia se usan
    más que su verbo.
    """
    pos, lema, _ = datos
    if DERIVADA.search(w):
        return False
    bases = {forma.stem for forma in diccionario.lookuper.good_forms(w)}
    if bases and all(INFINITIVO.search(b) and b != w for b in bases):
        # Lo que spaCy toma por sustantivo (espada, de espadar) solo cae si el verbo se usa más.
        if pos in ("VERB", "AUX", "PROPN"):
            return True
        return max(zipf_frequency(b, "es") for b in bases) > zipf_frequency(w, "es")
    if pos not in ("VERB", "AUX") or INFINITIVO.search(w):
        return False
    if nominal < 2:
        return True
    return lema != w and INFINITIVO.search(lema) and zipf_frequency(lema, "es") > zipf_frequency(w, "es")


def es_nombre_propio(nlp, w):
    """Madrid, Juan, Google…: la forma con mayúscula es mucho más frecuente que en minúscula.

    Pasa también con algunas comunes (el Ayuntamiento, la Universidad): esas se rescatan
    en tools/incluir.txt. La tabla de vectores está ordenada por frecuencia, así que la
    fila sirve de rango.
    """
    filas = nlp.vocab.vectors.key2row
    minuscula = filas.get(nlp.vocab.strings[w], -1)
    mayuscula = filas.get(nlp.vocab.strings.add(w.capitalize()), -1)
    return mayuscula >= 0 and mayuscula * 3 < minuscula


def lema_de(w, analisis, conjunto):
    """Clave para agrupar formas: el singular, y el masculino de los adjetivos.

    No se usa el lema de spaCy tal cual porque, sin contexto, convierte pregunta en
    preguntar o puerta en puerto; solo vale cuando es el singular (veces → vez).
    """
    if w in PLURALES:
        return w
    for corte in (2, 1):  # árboles → árbol, casas → casa
        if w.endswith("es"[-corte:]) and w[:-corte] in conjunto:
            return lema_de(w[:-corte], analisis, conjunto)
    pos, lema, _ = analisis[w]
    if pos == "ADJ":
        for fem, masc in (("as", "os"), ("a", "o")):
            if w.endswith(fem) and w[: -len(fem)] + masc in conjunto:
                return w[: -len(fem)] + masc
    if lema != w and w.endswith("s") and sin_tildes(lema) in {
        sin_tildes(w[:-1]), sin_tildes(w[:-2]), sin_tildes(w[:-3]) + "z"
    }:
        return lema
    return w


def sin_tildes(w):
    return w.translate(str.maketrans("áéíóúü", "aeiouu"))


def main():
    nlp = spacy.load("es_core_news_lg", exclude=["parser", "ner"])
    excluidas = leer_lista(EXCLUIR) - set(DIAS)
    incluidas = leer_lista(INCLUIR) - excluidas
    vocab, fundidos = construir_vocabulario(nlp, excluidas, incluidas)
    print(f"Vocabulario final: {len(vocab)} palabras", file=sys.stderr)

    matriz = np.array([nlp.vocab.get_vector(w) for w in vocab], dtype=np.float32)
    matriz /= np.linalg.norm(matriz, axis=1, keepdims=True)
    # Las palabras poco usadas tienen vectores menos fiables y tienden a colarse arriba:
    # pierden un poco de similitud por cada punto de frecuencia (zipf) por debajo de 3.
    rareza = np.array([max(0.0, 3.0 - zipf_frequency(w, "es")) for w in vocab], dtype=np.float32)
    indice = {w: i for i, w in enumerate(vocab)}

    DATOS.mkdir(exist_ok=True)
    (DATOS / "vocabulario.json").write_text(
        json.dumps(vocab, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )

    for num, palabra in enumerate(DIAS, start=1):
        i = indice[palabra]
        similitud = matriz @ matriz[i] - PENALIZACION_RARAS * rareza
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
