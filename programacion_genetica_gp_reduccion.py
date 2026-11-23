"""Programación genética multiárbol para descriptores de MNIST."""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, asdict
from functools import partial
from pathlib import Path

import csv
import hashlib
import json
import random
import shutil
import time
import urllib.request

import joblib
import numpy as np

from deap import base, creator, gp, tools

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from funciones_terminales_gp_reduccion import (
    construir_pset,
    construir_pset_reduccion,
    generar_expresion,
)
from red_convolucional_funcional_gp_reduccion import RedConvolucionalFuncional


@dataclass
class Configuracion:
    poblacion: int = 20
    generaciones: int = 10
    elitismo: int = 2
    torneo: int = 3

    prob_cruce: float = 0.8
    prob_mutacion: float = 0.3

    profundidad_inicial: int = 3
    profundidad_mutacion: int = 2

    altura_maxima: int = 6
    nodos_max_arbol: int = 60
    nodos_max_individuo: int = 300

    penalizacion_nodos: float = 1e-4

    svm_C: float = 10.0
    svm_kernel: str = "rbf"
    svm_gamma: str = "scale"

    cache_evaluaciones: int = 512
    semilla: int | None = None


if not hasattr(creator, "FitnessImagenesMin"):
    creator.create(
        "FitnessImagenesMin",
        base.Fitness,
        weights=(-1.0,),
    )

if not hasattr(creator, "IndividuoImagenes"):
    creator.create(
        "IndividuoImagenes",
        list,
        fitness=creator.FitnessImagenesMin,
    )


def contar_nodos(individuo):
    """Incluye primitivas y terminales de los diez árboles."""
    return sum(len(arbol) for arbol in individuo)


def clave_individuo(individuo):
    return tuple(str(arbol) for arbol in individuo)


def extraer_descriptores(imagenes, funciones):
    """
    Utiliza la clase anterior.

    MNIST:
        (28,28) -> (5,28,28) -> (5,14,14)
        -> (20,14,14) -> (20,7,7) -> (980,)
    """
    imagenes = np.asarray(imagenes)

    if imagenes.ndim != 3 or len(imagenes) == 0:
        raise ValueError("Se requiere un lote no vacío (N,H,W).")

    funciones_convolucionales, funcion_reduccion = funciones
    red = RedConvolucionalFuncional(funciones_convolucionales, funcion_reduccion)

    primer_descriptor = red.ejecutar(imagenes[0])

    X = np.empty(
        (len(imagenes), primer_descriptor.size),
        dtype=np.float32,
    )
    X[0] = primer_descriptor

    for i in range(1, len(imagenes)):
        X[i] = red.ejecutar(imagenes[i])

    if not np.isfinite(X).all():
        raise ValueError("El descriptor contiene valores no finitos.")

    return X


def crear_svm(configuracion):
    """
    StandardScaler se ajusta únicamente con los datos usados en fit.
    """
    return make_pipeline(
        StandardScaler(),
        SVC(
            C=configuracion.svm_C,
            kernel=configuracion.svm_kernel,
            gamma=configuracion.svm_gamma,
            cache_size=512,
        ),
    )

