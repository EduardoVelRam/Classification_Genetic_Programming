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


class ProgramacionGeneticaMultiArbol:

    def __init__(self, configuracion=None):
        self.config = configuracion or Configuracion()
        self._validar_configuracion()

        if self.config.semilla is None:
            self.config.semilla = time.time_ns() % (2**32 - 1)

        random.seed(self.config.semilla)

        self.pset = construir_pset()
        self.pset_reduccion = construir_pset_reduccion(n_canales=5)
        self.toolbox = base.Toolbox()

        self.toolbox.register("clone", deepcopy)
        self.toolbox.register("individual", self.crear_individuo)
        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.individual,
        )
        self.toolbox.register(
            "compile",
            gp.compile,
            pset=self.pset,
        )
        self.toolbox.register(
            "select",
            tools.selTournament,
            tournsize=self.config.torneo,
        )
        self.toolbox.register("mate", self.cruzar)
        self.toolbox.register("mutate", self.mutar)
        self.toolbox.register("evaluate", self.evaluar)

        self.cache = OrderedDict()
        self.historial = []
        self.mejor = None
        self.entrenamientos_svm = 0

    def _validar_configuracion(self):
        c = self.config

        if not 1 <= c.elitismo < c.poblacion:
            raise ValueError("Se requiere 1 <= elitismo < población.")

        if c.generaciones < 0 or c.torneo < 2:
            raise ValueError("Generaciones >=0 y torneo >=2.")

        if not (
            0 <= c.prob_cruce <= 1
            and 0 <= c.prob_mutacion <= 1
        ):
            raise ValueError("Las probabilidades deben estar en [0,1].")

        if not 1 <= c.profundidad_inicial <= c.altura_maxima:
            raise ValueError("Límites de profundidad incorrectos.")

        if (
            c.profundidad_mutacion < 0
            or c.nodos_max_arbol < 2
            or c.nodos_max_individuo < 9
            or c.cache_evaluaciones < 1
        ):
            raise ValueError("Límites de nodos o caché incorrectos.")

        if c.penalizacion_nodos < 0 or c.svm_C <= 0:
            raise ValueError("Penalización >=0 y C >0.")

    def dentro_limites(self, individuo):
        c = self.config

        return (
            len(individuo) == 10
            and contar_nodos(individuo) <= c.nodos_max_individuo
            and all(
                arbol.height <= c.altura_maxima
                and len(arbol) <= c.nodos_max_arbol
                for arbol in individuo
            )
        )

    def crear_individuo(self):
        """Nueve árboles imagen->imagen y un árbol de reducción 5->1."""
        for _ in range(200):
            arboles = [
                gp.PrimitiveTree(
                    generar_expresion(
                        self.pset,
                        min_=1,
                        max_=self.config.profundidad_inicial,
                    )
                )
                for _ in range(9)
            ]
            arboles.append(
                gp.PrimitiveTree(
                    generar_expresion(
                        self.pset_reduccion,
                        min_=1,
                        max_=self.config.profundidad_inicial,
                    )
                )
            )
            individuo = creator.IndividuoImagenes(arboles)

            if self.dentro_limites(individuo):
                return individuo

        raise ValueError(
            "No se pudo inicializar. Aumente los límites de nodos."
        )

    def compilar(self, individuo):
        """Compila nueve funciones 1->1 y una función reductora 5->1."""
        if len(individuo) != 10:
            raise ValueError("El individuo debe tener diez árboles.")
        funciones = [
            gp.compile(expr=arbol, pset=self.pset)
            for arbol in individuo[:9]
        ]
        funcion_reduccion = gp.compile(
            expr=individuo[9], pset=self.pset_reduccion
        )
        return funciones, funcion_reduccion

    def cruzar(self, a, b):
        """
        Cruza el mismo componente en ambos individuos.

        Esto conserva el papel de cada árbol, incluida la reducción.
        """
        indice = random.randrange(10)

        anterior_a = deepcopy(a[indice])
        anterior_b = deepcopy(b[indice])

        if random.random() < 0.25:
            # Intercambio completo del componente.
            a[indice], b[indice] = b[indice], a[indice]
        else:
            # Cruce de subárboles compatible con los tipos.
            gp.cxOnePoint(a[indice], b[indice])

        if not self.dentro_limites(a):
            a[indice] = anterior_a

        if not self.dentro_limites(b):
            b[indice] = anterior_b

        return a, b

    def mutar(self, individuo):
        """Muta un subárbol de cualquiera de los diez componentes."""
        indice = random.randrange(10)
        anterior = deepcopy(individuo[indice])

        generador = partial(
            generar_expresion,
            min_=0,
            max_=self.config.profundidad_mutacion,
        )

        pset = self.pset_reduccion if indice == 9 else self.pset
        gp.mutUniform(
            individuo[indice],
            expr=generador,
            pset=pset,
        )

        if not self.dentro_limites(individuo):
            individuo[indice] = anterior

        return individuo,

    def evaluar(self, individuo):
        """
        Minimiza:
            1 - exactitud_validacion + alpha * nodos_totales
        """
        clave = clave_individuo(individuo)

        if clave in self.cache:
            self.cache.move_to_end(clave)
            metricas = self.cache[clave]

        else:
            nodos = contar_nodos(individuo)
            penalizacion = self.config.penalizacion_nodos * nodos

            try:
                if not self.dentro_limites(individuo):
                    raise ValueError("Individuo fuera de límites.")

                funciones = self.compilar(individuo)

                X_train = extraer_descriptores(
                    self.imagenes_train,
                    funciones,
                )
                X_valid = extraer_descriptores(
                    self.imagenes_valid,
                    funciones,
                )

                clasificador = crear_svm(self.config)
                clasificador.fit(X_train, self.y_train)

                self.entrenamientos_svm += 1

                predicciones = clasificador.predict(X_valid)
                exactitud = float(
                    accuracy_score(self.y_valid, predicciones)
                )

                metricas = {
                    "exactitud_validacion": exactitud,
                    "nodos": nodos,
                    "penalizacion": penalizacion,
                    "aptitud": 1.0 - exactitud + penalizacion,
                    "error": "",
                }

            except (ValueError, ArithmeticError, RuntimeError) as exc:
                metricas = {
                    "exactitud_validacion": 0.0,
                    "nodos": nodos,
                    "penalizacion": penalizacion,
                    "aptitud": 10.0 + penalizacion,
                    "error": f"{type(exc).__name__}: {exc}",
                }

            self.cache[clave] = metricas

            if len(self.cache) > self.config.cache_evaluaciones:
                self.cache.popitem(last=False)

        individuo.metricas = dict(metricas)

        return metricas["aptitud"],

    def _evaluar_invalidos(self, poblacion):
        for individuo in poblacion:
            if not individuo.fitness.valid:
                individuo.fitness.values = self.toolbox.evaluate(
                    individuo
                )

    def _registrar(self, poblacion, generacion):
        mejor = tools.selBest(poblacion, 1)[0]

        registro = {
            "generacion": generacion,
            "aptitud_min": mejor.fitness.values[0],
            "aptitud_media": float(
                np.mean(
                    [i.fitness.values[0] for i in poblacion]
                )
            ),
            "exactitud_mejor": mejor.metricas[
                "exactitud_validacion"
            ],
            "nodos_mejor": contar_nodos(mejor),
            "evaluaciones_fallidas": sum(
                bool(i.metricas["error"])
                for i in poblacion
            ),
            "entrenamientos_svm": self.entrenamientos_svm,
        }

        self.historial.append(registro)

        print(
            f"G{generacion:03d} | "
            f"aptitud={registro['aptitud_min']:.6f} | "
            f"accuracy={registro['exactitud_mejor']:.4f} | "
            f"nodos={registro['nodos_mejor']} | "
            f"fallos={registro['evaluaciones_fallidas']}",
            flush=True,
        )

    def evolucionar(
        self,
        imagenes_train,
        y_train,
        imagenes_valid,
        y_valid,
    ):
        """
        La función no recibe el conjunto de prueba.

        Las particiones deben permanecer fijas durante la evolución.
        """
        self.imagenes_train = np.asarray(imagenes_train)
        self.imagenes_valid = np.asarray(imagenes_valid)
        self.y_train = np.asarray(y_train)
        self.y_valid = np.asarray(y_valid)

        if (
            len(self.imagenes_train) != len(self.y_train)
            or len(self.imagenes_valid) != len(self.y_valid)
            or len(np.unique(self.y_train)) < 2
            or len(self.y_valid) == 0
        ):
            raise ValueError("Particiones o etiquetas incorrectas.")

        self.cache.clear()
        self.historial = []
        self.entrenamientos_svm = 0
        self.mejor = None

        poblacion = self.toolbox.population(
            n=self.config.poblacion
        )

        # Referencia: identidad en convoluciones y suma de los cinco mapas.
        identidad = [
            gp.PrimitiveTree.from_string("I", self.pset)
            for _ in range(9)
        ]
        identidad.append(
            gp.PrimitiveTree.from_string(
                "suma(suma(C0, C1), suma(suma(C2, C3), C4))",
                self.pset_reduccion,
            )
        )
        poblacion[0] = creator.IndividuoImagenes(identidad)

        self._evaluar_invalidos(poblacion)

        if all(i.metricas["error"] for i in poblacion):
            raise RuntimeError(
                "Todas las evaluaciones fallaron: "
                + poblacion[0].metricas["error"]
            )

        salon_fama = tools.HallOfFame(1)
        salon_fama.update(poblacion)

        self._registrar(poblacion, 0)

        for generacion in range(
            1, self.config.generaciones + 1
        ):
            # Copias intactas: los élites no reciben variación.
            elites = [
                self.toolbox.clone(individuo)
                for individuo in tools.selBest(
                    poblacion,
                    self.config.elitismo,
                )
            ]

            seleccionados = self.toolbox.select(
                poblacion,
                len(poblacion) - len(elites),
            )

            hijos = [
                self.toolbox.clone(individuo)
                for individuo in seleccionados
            ]

            for a, b in zip(hijos[::2], hijos[1::2]):
                if random.random() < self.config.prob_cruce:
                    self.toolbox.mate(a, b)
                    del a.fitness.values
                    del b.fitness.values

            for hijo in hijos:
                if random.random() < self.config.prob_mutacion:
                    self.toolbox.mutate(hijo)
                    del hijo.fitness.values

            self._evaluar_invalidos(hijos)

            poblacion = elites + hijos

            salon_fama.update(poblacion)
            self._registrar(poblacion, generacion)

        self.mejor = self.toolbox.clone(salon_fama[0])

        return self.mejor

    def evaluar_prueba(self, imagenes_test, y_test):
        """
        Después de seleccionar el mejor individuo:

        1. Reajusta escalador y SVM con train + valid.
        2. Evalúa el conjunto de prueba.
        """
        if self.mejor is None:
            raise RuntimeError("Primero ejecute evolucionar.")

        funciones = self.compilar(self.mejor)

        imagenes_ajuste = np.concatenate(
            [self.imagenes_train, self.imagenes_valid]
        )
        y_ajuste = np.concatenate(
            [self.y_train, self.y_valid]
        )

        clasificador = crear_svm(self.config)

        clasificador.fit(
            extraer_descriptores(imagenes_ajuste, funciones),
            y_ajuste,
        )

        predicciones = clasificador.predict(
            extraer_descriptores(imagenes_test, funciones)
        )

        etiquetas = np.unique(
            np.concatenate([y_ajuste, y_test])
        )

        resultados = {
            "exactitud_test": float(
                accuracy_score(y_test, predicciones)
            ),
            "etiquetas": etiquetas.tolist(),
            "matriz_confusion": confusion_matrix(
                y_test,
                predicciones,
                labels=etiquetas,
            ).tolist(),
            "reporte": classification_report(
                y_test,
                predicciones,
                labels=etiquetas,
                output_dict=True,
                zero_division=0,
            ),
        }

        return clasificador, resultados, predicciones

    def guardar(self, carpeta, clasificador, resultados):
        carpeta = Path(carpeta)
        carpeta.mkdir(parents=True, exist_ok=True)

        documento = {
            "configuracion": asdict(self.config),
            "arboles": list(clave_individuo(self.mejor)),
            "nodos_por_arbol": [
                len(arbol) for arbol in self.mejor
            ],
            "metricas_evolucion": self.mejor.metricas,
            "resultados_prueba": resultados,
        }

        (carpeta / "mejor_individuo.json").write_text(
            json.dumps(
                documento,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        with (carpeta / "historial.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as archivo:
            escritor = csv.DictWriter(
                archivo,
                fieldnames=self.historial[0].keys(),
            )
            escritor.writeheader()
            escritor.writerows(self.historial)

        joblib.dump(
            clasificador,
            carpeta / "svm.joblib",
            compress=3,
        )


# ------------------------------------------------------------------
# MNIST: descarga directa, sin TensorFlow ni Keras
# ------------------------------------------------------------------

URL_MNIST = (
    "https://storage.googleapis.com/"
    "tensorflow/tf-keras-datasets/mnist.npz"
)
SHA256_MNIST = (
    "731c5ac602752760c8e48fbffcf8c3b850"
    "d9dc2a2aedcf2cc48468fc17b673d1"
)


def cargar_mnist(ruta="datos/mnist.npz"):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    if not ruta.exists():
        temporal = ruta.with_suffix(".download")

        try:
            print("Descargando MNIST...", flush=True)

            with urllib.request.urlopen(
                URL_MNIST, timeout=60
            ) as entrada:
                with temporal.open("wb") as salida:
                    shutil.copyfileobj(entrada, salida)

            digest = hashlib.sha256(
                temporal.read_bytes()
            ).hexdigest()

            if digest != SHA256_MNIST:
                raise ValueError("SHA256 de descarga incorrecto.")

            temporal.replace(ruta)

        finally:
            temporal.unlink(missing_ok=True)

    digest = hashlib.sha256(ruta.read_bytes()).hexdigest()

    if digest != SHA256_MNIST:
        raise ValueError("El archivo MNIST local no es el esperado.")

    with np.load(ruta, allow_pickle=False) as datos:
        x_train = datos["x_train"]
        y_train = datos["y_train"].astype(np.int64)
        x_test = datos["x_test"]
        y_test = datos["y_test"].astype(np.int64)

    if (
        x_train.shape != (60000, 28, 28)
        or x_test.shape != (10000, 28, 28)
        or y_train.shape != (60000,)
        or y_test.shape != (10000,)
    ):
        raise ValueError("Las dimensiones de MNIST son incorrectas.")

    return x_train, y_train, x_test, y_test


def preparar_particiones(
    x,
    y,
    x_test,
    y_test,
    n_train,
    n_valid,
    n_test,
    semilla,
):
    """
    Entrenamiento y validación proceden del bloque oficial de 60000.
    Prueba procede exclusivamente del bloque oficial de 10000.
    """
    if (
        n_train < 10
        or n_valid < 10
        or n_train + n_valid > len(y)
        or not 10 <= n_test <= len(y_test)
    ):
        raise ValueError("Tamaños de partición incorrectos.")

    indices_train, indices_valid = train_test_split(
        np.arange(len(y)),
        train_size=n_train,
        test_size=n_valid,
        stratify=y,
        random_state=semilla,
    )

    if n_test == len(y_test):
        indices_test = np.arange(len(y_test))
    else:
        indices_test, _ = train_test_split(
            np.arange(len(y_test)),
            train_size=n_test,
            stratify=y_test,
            random_state=(semilla + 1) % (2**32 - 1),
        )

    normalizar = lambda a: a.astype(np.float32) / 255.0

    return {
        "x_train": normalizar(x[indices_train]),
        "y_train": y[indices_train],

        "x_valid": normalizar(x[indices_valid]),
        "y_valid": y[indices_valid],

        "x_test": normalizar(x_test[indices_test]),
        "y_test": y_test[indices_test],

        "indices_train": indices_train,
        "indices_valid": indices_valid,
        "indices_test": indices_test,
    }

