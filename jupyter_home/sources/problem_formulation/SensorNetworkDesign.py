import math
from typing import Tuple

import numpy as np
from numpy.linalg import matrix_rank, inv
from numpy.matlib import eye
from scipy.linalg import qr


class SensorNetworkDesign:
    """
    This class defines a Sensor Network Design Problem and evaluates a proposed solution
    """

    def __init__(self, costo: np.array, incidence_matrix: np.array, absolute_precision_matrix: np.array,
                 required: np.array, bounds: np.array):

        """

        :param costo: vector de coste de los componentes para medir cada flujo
        :param incidence_matrix: matrix con el sistema de ecuaciones del balance de masa (lo que llaman d en los
        ejemplos de matlab de mercedes).
        :param absolute_precision_matrix: vector con la precision que tiene el componente que medirá un flujo
        :param required: flujos del sistema que nos importa medir.
        :param bounds: precision minima con la que queremos medir los flujos del sistema.
        """

        self.costo = costo
        self.incidence_matrix = incidence_matrix
        self.absolute_precision_matrix = absolute_precision_matrix
        self.required = required
        self.bounds = bounds

        """
        La matriz de dependencias tiene en la primera fila los flujos del sistema que se necesitan medir y en la segunda
        la precesion minima con la que queremos medirlos.
        """
        n_required = required.size
        self.dependence_matrix = np.zeros((2, n_required))
        self.dependence_matrix[0, :] = required
        self.dependence_matrix[1, :] = self.bounds[required]

    def dimensions(self):
        return self.costo.shape[0]

    def analyze_solution(self, flows_measured: np.array) -> Tuple[np.array, int, np.array, np.array]:
        """
        No esta programada de una forma muy clara ya que he traducido el codigo de matlab tal cual, he hecho pruebas
        de caja negra con las soluciones de matlab y las soluciones de esta funcion para unos 200 ejemplos y funciona
        igual con una covertura del 97% de las lineas. Hay dos if por los que no pasan ningun ejemplo serán algún caso
        extremo...

        :param flows_measured: un vector binario donde un 1 indica que hay un sensor que mide ese flujo y un 0 que no
        lo hay. Debe tener el mismo tamaño que el vector de coste y la segunda dimension de la matrix 'incidence_matrix'

        :return: devuelte una tupla de cuatro componentes: el primero es la precision para cada flujo que se quiere
        medir del sistema; el segundo es un 1 si la solución cumple todas las restricciones y un 0 si no las cumple; el
        tercero es un vector binario, del tamaño del número de flujos que se quieren medir, dónde un 0 significa que ese
        flujo no es observable y un 1 indica que si lo es; el cuarto es un vector binario, del tamaño del número de
        flujos que se quieren medir donde un 0 indica que ese flujo no cumple las restricciones de precision impuestas y
        un 1 indica que si las cumple.
        """
        n_flows = flows_measured.shape[0]

        measured_flows = np.argwhere(flows_measured == 1)
        not_measured_flows = np.argwhere(flows_measured == 0)

        deviation_flows = self.absolute_precision_matrix[measured_flows][:, 0]
        Q = np.diag(deviation_flows**2)

        """
        Clasificacion de variables
         1: medida redundantes        2: medida no redundante
         3: no medida  observable        4: no medida unobservable
        """
        classes = np.zeros(n_flows)
        allnr = False

        measured_incidences = self.incidence_matrix[:, measured_flows][:, :, 0]
        if len(measured_flows) != n_flows:
            not_measured_incidences = self.incidence_matrix[:, not_measured_flows][:, :, 0]

            QB, RB, iu = qr(not_measured_incidences, pivoting=True)

            ru = matrix_rank(RB)
            mu, nu = not_measured_incidences.shape

            QB1 = QB[:, 0:ru]
            QB2 = QB[:, ru:mu]
            RB1 = RB[0:ru, 0:ru]
            RB2 = RB[0:ru, ru:nu]

            ur = iu[0:ru]
            unr = iu[ru:nu]
            RI = np.dot(inv(RB1), RB2)

            ind = np.zeros(ru)
            for k in range(ru):
                if np.all(abs(RI[k, :]) < 0.001):
                    ind[k] = 1
                else:
                    ind[k] = 0

            if nu == ru:
                classes[not_measured_flows[:, 0]] = 3*np.ones(nu)
            else:
                classes[not_measured_flows[unr, 0]] = 4*np.ones(nu-ru)
                for k in range(ru):
                    if ind[k] == 1:
                        classes[int(not_measured_flows[ur[k], 0])] = 3
                    else:
                        classes[int(not_measured_flows[ur[k], 0])] = 4
        else:
            QB = eye(self.incidence_matrix.shape[0])
            mu = self.incidence_matrix.shape[0]
            ru = 0

            QB1 = QB[:, 0:ru]
            QB2 = QB[:, ru:mu]

        G = np.dot(np.transpose(QB2), measured_incidences)
        lx = measured_flows.shape[0]

        if ru == mu:
            classes[measured_flows[:, 0]] = 2*np.ones(lx)
            allnr = True
        else:
            for k in range(lx):
                if np.all(abs(G[:, k]) < 0.001):
                    classes[int(measured_flows[k, 0])] = 2
                else:
                    classes[int(measured_flows[k, 0])] = 1

        """
        Se inicia la verificación de precision
        """
        sal = 1
        ndp = self.dependence_matrix.shape[1]
        salob = np.ones(ndp)
        precisions_met = np.ones(ndp)
        p = np.zeros(ndp)

        for k in range(ndp):
            if classes[int(self.dependence_matrix[0, k])] == 4:
                salob[k] = 0
                precisions_met[k] = 0
                sal = 0

        """
        SI SAL!= 0 TODAS LAS VARIABLES CON RESTRICCIONES DE OBSERVABILIDAD SON OBSERVABLES
        Calcular los desvios estandar de las estimaciones luego de aplicar la tecnica de reconciliacion de datos
        """
        if sal != 0:

            if allnr is True:
                Sx = Q
                Z = eye(lx)
            else:
                J = np.dot(np.dot(G, Q), np.transpose(G))
                I = eye(lx)
                Z = I - np.dot(np.dot(np.dot(Q, np.transpose(G)), inv(J)), G)
                Sx = np.dot(np.dot(Z, Q), np.transpose(Z))

            # Desvio standard de las medidas
            dsx = np.sqrt(np.diag(Sx))

            if len(measured_flows) != n_flows:
                H = np.dot(np.dot(np.dot(inv(RB1), np.transpose(QB1)), measured_incidences), Z)
                Su = np.dot(np.dot(H, Q), np.transpose(H))

                dsu = np.zeros(ru)
                for k in range(ru):
                    if ind[k] == 1:
                        # desvio standard de las no medidas
                        dsu[k] = math.sqrt(Su[k, k])
                    else:
                        dsu[k] = 0

            for k in range(ndp):
                if self.dependence_matrix[0, k] in measured_flows:
                    index = np.where(measured_flows == self.dependence_matrix[0, k])[0][0]
                    p[k] = dsx[index]
                else:
                    for w in range(ru):
                        found = False
                        if not found and ind[w] == 1 and not_measured_flows[ur[w]] == self.dependence_matrix[0, k]:
                            p[k] = dsu[w]
                            found = True

            """
            Verificar si se satisfacen las restricciones de precision
            """
            for k in range(ndp):
                if p[k] > self.dependence_matrix[1, k]:
                    sal = 0
                    precisions_met[k] = 0

        return p, sal, salob, precisions_met







