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

