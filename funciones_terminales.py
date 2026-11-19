"""Funciones y terminales externos para la programación genética."""

import random
from functools import partial

import numpy as np
from scipy import ndimage as ndi
from deap import gp


IMAGEN = np.ndarray
EPS = 1e-6
LIMITE = 10.0


def proteger(imagen):
    """Conserva números finitos y acota la salida de cada operador."""
    imagen = np.asarray(imagen, dtype=np.float32)
    imagen = np.nan_to_num(
        imagen,
        nan=0.0,
        posinf=LIMITE,
        neginf=-LIMITE,
    )
    return np.clip(imagen, -LIMITE, LIMITE)


# ------------------------------------------------------------------
# Operaciones aritméticas entre imágenes
# ------------------------------------------------------------------

def suma(a, b):
    return proteger(proteger(a) + proteger(b))


def resta(a, b):
    return proteger(proteger(a) - proteger(b))


def producto(a, b):
    return proteger(proteger(a) * proteger(b))


def division(a, b):
    a, b = proteger(a), proteger(b)
    denominador = np.where(
        np.abs(b) < EPS,
        np.where(b < 0, -EPS, EPS),
        b,
    )
    return proteger(a / denominador)


# ------------------------------------------------------------------
# Operaciones puntuales
# ------------------------------------------------------------------

def logaritmo(a):
    return proteger(np.log(np.abs(proteger(a)) + EPS))


def exponencial(a):
    return proteger(np.exp(np.clip(proteger(a), -5.0, 2.0)))


def seno(a):
    return proteger(np.sin(proteger(a)))


def coseno(a):
    return proteger(np.cos(proteger(a)))


def tanh(a):
    return proteger(np.tanh(proteger(a)))


def absoluto(a):
    return proteger(np.abs(proteger(a)))


def raiz(a):
    return proteger(np.sqrt(np.abs(proteger(a))))


def negativo(a):
    return proteger(-proteger(a))


def constante_imagen(a, valor):
    """Crea una imagen constante con las dimensiones de a."""
    return proteger(np.full_like(a, valor, dtype=np.float32))


# ------------------------------------------------------------------
# Filtros gaussianos y derivadas
# ------------------------------------------------------------------

def sigma_segura(sigma):
    return float(np.clip(abs(sigma), 0.5, 3.0))


def gaussiano(a, sigma):
    return proteger(
        ndi.gaussian_filter(
            proteger(a),
            sigma=sigma_segura(sigma),
            mode="reflect",
            truncate=3.0,
        )
    )


def derivada_gaussiana(a, sigma, orden):
    """
    Ejes NumPy: (y, x).

    Gx  -> (0, 1)
    Gy  -> (1, 0)
    Gxx -> (0, 2)
    Gyy -> (2, 0)
    Gxy -> (1, 1)
    """
    return proteger(
        ndi.gaussian_filter(
            proteger(a),
            sigma=sigma_segura(sigma),
            order=orden,
            mode="reflect",
            truncate=3.0,
        )
    )


def laplaciano(a):
    return proteger(ndi.laplace(proteger(a), mode="reflect"))


def laplaciano_gaussiano(a, sigma):
    return proteger(
        ndi.gaussian_laplace(
            proteger(a),
            sigma=sigma_segura(sigma),
            mode="reflect",
            truncate=3.0,
        )
    )


# ------------------------------------------------------------------
# Sobel, Prewitt y Harris
# ------------------------------------------------------------------

def sobel_x(a):
    return proteger(ndi.sobel(proteger(a), axis=1, mode="reflect"))


def sobel_y(a):
    return proteger(ndi.sobel(proteger(a), axis=0, mode="reflect"))


def prewitt_x(a):
    return proteger(ndi.prewitt(proteger(a), axis=1, mode="reflect"))


def prewitt_y(a):
    return proteger(ndi.prewitt(proteger(a), axis=0, mode="reflect"))


def magnitud_gradiente(a):
    return proteger(np.hypot(sobel_x(a), sobel_y(a)))


def harris(a):
    """
    Devuelve el mapa de respuesta Harris completo.

    No aplica umbralización ni supresión de no máximos.
    """
    a = proteger(a)

    ix = ndi.gaussian_filter(
        a, sigma=1.0, order=(0, 1), mode="reflect"
    )
    iy = ndi.gaussian_filter(
        a, sigma=1.0, order=(1, 0), mode="reflect"
    )

    m11 = ndi.gaussian_filter(ix * ix, sigma=1.0, mode="reflect")
    m22 = ndi.gaussian_filter(iy * iy, sigma=1.0, mode="reflect")
    m12 = ndi.gaussian_filter(ix * iy, sigma=1.0, mode="reflect")

    respuesta = m11 * m22 - m12**2 - 0.04 * (m11 + m22)**2

    return proteger(respuesta)


# ------------------------------------------------------------------
# Morfología en escala de grises
# ------------------------------------------------------------------

def morfologia(a, operacion):
    a = proteger(a)

    dilatar = lambda x: ndi.grey_dilation(
        x, size=(3, 3), mode="reflect"
    )
    erosionar = lambda x: ndi.grey_erosion(
        x, size=(3, 3), mode="reflect"
    )
    abrir = lambda x: ndi.grey_opening(
        x, size=(3, 3), mode="reflect"
    )
    cerrar = lambda x: ndi.grey_closing(
        x, size=(3, 3), mode="reflect"
    )

    operaciones = {
        "dilatacion": lambda: dilatar(a),
        "erosion": lambda: erosionar(a),
        "apertura": lambda: abrir(a),
        "cierre": lambda: cerrar(a),
        "gradiente_morfologico": lambda: dilatar(a) - erosionar(a),
        "tophat": lambda: a - abrir(a),
        "blackhat": lambda: cerrar(a) - a,
    }

    return proteger(operaciones[operacion]())


# ------------------------------------------------------------------
# Catálogo: nombre, función, tipos de argumentos
# ------------------------------------------------------------------

PRIMITIVAS = [
    ("suma", suma, [IMAGEN, IMAGEN]),
    ("resta", resta, [IMAGEN, IMAGEN]),
    ("producto", producto, [IMAGEN, IMAGEN]),
    ("division", division, [IMAGEN, IMAGEN]),

    ("log", logaritmo, [IMAGEN]),
    ("exp", exponencial, [IMAGEN]),
    ("sin", seno, [IMAGEN]),
    ("cos", coseno, [IMAGEN]),
    ("tanh", tanh, [IMAGEN]),
    ("abs", absoluto, [IMAGEN]),
    ("raiz", raiz, [IMAGEN]),
    ("neg", negativo, [IMAGEN]),

    ("constante", constante_imagen, [IMAGEN, float]),
    ("gauss", gaussiano, [IMAGEN, float]),

    (
        "Gx",
        partial(derivada_gaussiana, orden=(0, 1)),
        [IMAGEN, float],
    ),
    (
        "Gy",
        partial(derivada_gaussiana, orden=(1, 0)),
        [IMAGEN, float],
    ),
    (
        "Gxx",
        partial(derivada_gaussiana, orden=(0, 2)),
        [IMAGEN, float],
    ),
    (
        "Gyy",
        partial(derivada_gaussiana, orden=(2, 0)),
        [IMAGEN, float],
    ),
    (
        "Gxy",
        partial(derivada_gaussiana, orden=(1, 1)),
        [IMAGEN, float],
    ),

    ("laplace", laplaciano, [IMAGEN]),
    ("LoG", laplaciano_gaussiano, [IMAGEN, float]),

    ("sobel_x", sobel_x, [IMAGEN]),
    ("sobel_y", sobel_y, [IMAGEN]),
    ("prewitt_x", prewitt_x, [IMAGEN]),
    ("prewitt_y", prewitt_y, [IMAGEN]),
    ("magnitud", magnitud_gradiente, [IMAGEN]),
    ("harris", harris, [IMAGEN]),
]

for nombre in (
    "dilatacion",
    "erosion",
    "apertura",
    "cierre",
    "gradiente_morfologico",
    "tophat",
    "blackhat",
):
    PRIMITIVAS.append(
        (
            nombre,
            partial(morfologia, operacion=nombre),
            [IMAGEN],
        )
    )


TERMINALES_ESCALARES = (
    -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0
)


def constante_aleatoria():
    return random.uniform(-3.0, 3.0)


def construir_pset():
    """
    Entrada: imagen I.
    Salida: imagen del mismo tamaño.

    Los escalares sirven como parámetros y para crear imágenes constantes.
    """
    pset = gp.PrimitiveSetTyped(
        "FUNCION_IMAGEN",
        [IMAGEN],
        IMAGEN,
    )
    pset.renameArguments(ARG0="I")

    for nombre, funcion, argumentos in PRIMITIVAS:
        pset.addPrimitive(
            funcion,
            argumentos,
            IMAGEN,
            name=nombre,
        )

    for valor in TERMINALES_ESCALARES:
        pset.addTerminal(valor, float)

    pset.addEphemeralConstant(
        "c",
        constante_aleatoria,
        float,
    )

    return pset


def generar_expresion(pset, min_=1, max_=3, type_=None):
    """
    Generador tipado compatible con parámetros float.

    Las ramas float terminan inmediatamente en una constante.
    Así no se solicitan primitivas escalares inexistentes.
    """
    altura = random.randint(min_, max_)
    tipo_raiz = pset.ret if type_ is None else type_

    expresion = []
    pendientes = [(0, tipo_raiz)]

    while pendientes:
        profundidad, tipo = pendientes.pop()

        terminar = (
            tipo is float
            or profundidad >= altura
            or (
                profundidad >= min_
                and random.random() < 0.25
            )
        )

        if terminar:
            terminal = random.choice(pset.terminals[tipo])

            if isinstance(terminal, gp.MetaEphemeral):
                terminal = terminal()

            expresion.append(terminal)
        else:
            primitiva = random.choice(pset.primitives[tipo])
            expresion.append(primitiva)

            pendientes.extend(
                (profundidad + 1, argumento)
                for argumento in reversed(primitiva.args)
            )

    return expresion
