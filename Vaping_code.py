import unicodedata
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve, brentq
from matplotlib.colors import LinearSegmentedColormap


def nombre_archivo_seguro(texto):
    """
    Normaliza un nombre de escenario para usarlo en un nombre de archivo:
    sin tildes, en minúsculas y con guiones bajos en lugar de espacios.

    Solo afecta los archivos generados; los títulos de las figuras conservan
    el nombre original con su acentuación.
    """
    sin_tildes = ''.join(c for c in unicodedata.normalize('NFD', texto)
                         if unicodedata.category(c) != 'Mn')
    return sin_tildes.lower().replace(' ', '_')


class ModeloVapeo:
    """
    Implementación completa del modelo matemático de vapeo.
    Basado en el sistema de ecuaciones diferenciales desarrollado en la tesis.
    """
    
    def __init__(self, params):
        """
        Inicializa el modelo con los parámetros calibrados.
        
        Parámetros:
        -----------
        params : dict
            Diccionario con los parámetros del modelo:
            - beta: tasa de transmisión para susceptibles
            - beta_p: tasa de transmisión para predispuestos
            - gamma_t: tasa de abandono temporal
            - gamma_p: tasa de abandono permanente
            - rho: tasa de recaída
            - phi: factor de reducción para susceptibles
            - phi_p: factor de reducción para predispuestos
            - q: proporción de susceptibles
            - mu: tasa de mortalidad
            - N: población total
        """
        self.params = params
        self.beta = params['beta']
        self.beta_p = params['beta_p']
        self.gamma_t = params['gamma_t']
        self.gamma_p = params['gamma_p']
        self.rho = params['rho']
        self.phi = params['phi']
        self.phi_p = params['phi_p']
        self.q = params['q']
        self.mu = params['mu']
        self.N = params['N']
        
    def calcular_R0(self):
        """
        Calcula el número reproductivo básico R_0.
        
        R_0 = (φ*β*q + φ_p*β_p*(1-q)) / (μ + γ_t + γ_p)
        
        Returns:
        --------
        float: Valor de R_0
        """
        numerador = self.phi * self.beta * self.q + self.phi_p * self.beta_p * (1 - self.q)
        denominador = self.mu + self.gamma_t + self.gamma_p
        R0 = numerador / denominador
        
        return R0
    
    def equilibrio_libre_vapeo(self):
        """
        Calcula el equilibrio libre de vapeo.
        
        Returns:
        --------
        tuple: (S*, P*, V*, Qt*, Qp*) = (qN, (1-q)N, 0, 0, 0)
        """
        S_star = self.q * self.N
        P_star = (1 - self.q) * self.N
        V_star = 0
        Qt_star = 0
        Qp_star = 0
        
        return (S_star, P_star, V_star, Qt_star, Qp_star)
    
    def coeficientes_cubica(self):
        """
        Calcula los coeficientes de la ecuación cúbica para V*.
        
        a_3*(V*)^3 + a_2*(V*)^2 + a_1*V* + a_0 = 0
        
        Returns:
        --------
        tuple: (a_3, a_2, a_1, a_0)
        """
        phi = self.phi
        beta = self.beta
        phi_p = self.phi_p
        beta_p = self.beta_p
        rho = self.rho
        mu = self.mu
        gamma_t = self.gamma_t
        gamma_p = self.gamma_p
        q = self.q
        N = self.N
        
        # Coeficiente a_3
        a_3 = phi * beta * phi_p * beta_p * rho * (mu + gamma_p)
        
        # Coeficiente a_2
        a_2 = mu * N * (
            rho * (phi * beta + phi_p * beta_p) * (mu + gamma_p) +
            phi * beta * phi_p * beta_p * (mu + gamma_t + gamma_p - rho)
        )
        
        # Coeficiente a_1
        a_1 = mu**2 * N**2 * (
            mu * (rho + phi * beta + phi_p * beta_p) +
            gamma_t * (phi * beta + phi_p * beta_p) +
            gamma_p * (rho + phi * beta + phi_p * beta_p) -
            phi * beta * phi_p * beta_p -
            rho * (phi * beta * q + phi_p * beta_p * (1 - q))
        )
        
        # Coeficiente a_0
        R0 = self.calcular_R0()
        a_0 = mu**3 * N**3 * (mu + gamma_t + gamma_p) * (1 - R0)
        
        return (a_3, a_2, a_1, a_0)
    
    def discriminante_cubica(self):
        """
        Calcula el discriminante de la ecuación cúbica.
        
        Para ax³ + bx² + cx + d = 0:
        Δ = 18abcd - 4b³d + b²c² - 4ac³ - 27a²d²
        
        Returns:
        --------
        float: Valor del discriminante
        """
        a, b, c, d = self.coeficientes_cubica()
        
        Delta = (18 * a * b * c * d - 
                 4 * b**3 * d + 
                 b**2 * c**2 - 
                 4 * a * c**3 - 
                 27 * a**2 * d**2)
        
        return Delta
    
    def puntos_criticos_cubica(self):
        """
        Calcula los puntos críticos de f(V*) = a_3*(V*)^3 + a_2*(V*)^2 + a_1*V* + a_0.
        
        Resuelve f'(V*) = 3*a_3*(V*)^2 + 2*a_2*V* + a_1 = 0
        
        Returns:
        --------
        tuple: (discriminante_derivada, puntos_criticos)
        """
        a_3, a_2, a_1, a_0 = self.coeficientes_cubica()
        
        Delta_prima = 4 * (a_2**2 - 3 * a_3 * a_1)
        
        puntos_criticos = []
        if Delta_prima > 0:
            sqrt_term = np.sqrt(Delta_prima)
            V_crit_1 = (-a_2 + sqrt_term / 2) / (3 * a_3)
            V_crit_2 = (-a_2 - sqrt_term / 2) / (3 * a_3)
            puntos_criticos = [V_crit_1, V_crit_2]
        elif Delta_prima == 0:
            V_crit = -a_2 / (3 * a_3)
            puntos_criticos = [V_crit]
        
        return (Delta_prima, puntos_criticos)
    
    def resolver_equilibrios_endemicos(self):
        """
        Resuelve la ecuación cúbica para encontrar todos los equilibrios endémicos.
        
        Returns:
        --------
        list: Lista de valores de V* positivos (equilibrios biológicamente relevantes)
        """
        a_3, a_2, a_1, a_0 = self.coeficientes_cubica()
        
        coeficientes = [a_3, a_2, a_1, a_0]
        raices = np.roots(coeficientes)
        
        raices_reales = raices[np.isreal(raices)].real
        raices_positivas = raices_reales[(raices_reales > 1e-10) & (raices_reales <= self.N)]
        
        return sorted(raices_positivas)
    
    def calcular_equilibrio_completo(self, V_star):
        """
        Dado V*, calcula todos los compartimentos del equilibrio endémico.
        
        Parameters:
        -----------
        V_star : float
            Valor de vapeadores en equilibrio
            
        Returns:
        --------
        dict: Diccionario con todos los compartimentos del equilibrio
        """
        # Validación de rango biológico
        if V_star < 0:
            raise ValueError(f"V* = {V_star:.4f} es negativo, fuera de rango biológico.")
        if V_star > self.N:
            raise ValueError(f"V* = {V_star:.4f} excede la población total N = {self.N}.")
        
        # Calcular S*
        S_star = (self.q * self.mu * self.N) / (self.phi * self.beta * V_star / self.N + self.mu)
        
        # Calcular P*
        P_star = ((1 - self.q) * self.mu * self.N) / (self.phi_p * self.beta_p * V_star / self.N + self.mu)
        
        # Calcular Qt*
        Qt_star = (self.gamma_t * V_star) / (self.rho * V_star / self.N + self.mu)
        
        # Calcular Qp*
        Qp_star = (self.gamma_p * V_star) / self.mu
        
        # Verificar conservación de población
        suma_total = S_star + P_star + V_star + Qt_star + Qp_star
        
        return {
            'S': S_star,
            'P': P_star,
            'V': V_star,
            'Qt': Qt_star,
            'Qp': Qp_star,
            'Total': suma_total,
            'Error_conservacion': abs(suma_total - self.N)
        }
    
    def matriz_jacobiana(self, equilibrio):
        """
        Calcula la matriz jacobiana del sistema evaluada en un equilibrio.
        
        Parameters:
        -----------
        equilibrio : tuple
            (S, P, V, Qt, Qp) valores del equilibrio
            
        Returns:
        --------
        numpy.ndarray: Matriz jacobiana 5x5
        """
        S, P, V, Qt, Qp = equilibrio
        
        J = np.zeros((5, 5))
        
        # Primera fila: dS/dt
        J[0, 0] = -self.phi * self.beta * V / self.N - self.mu
        J[0, 2] = -self.phi * self.beta * S / self.N
        
        # Segunda fila: dP/dt
        J[1, 1] = -self.phi_p * self.beta_p * V / self.N - self.mu
        J[1, 2] = -self.phi_p * self.beta_p * P / self.N
        
        # Tercera fila: dV/dt
        J[2, 0] = self.phi * self.beta * V / self.N
        J[2, 1] = self.phi_p * self.beta_p * V / self.N
        J[2, 2] = (self.phi * self.beta * S / self.N + 
                   self.phi_p * self.beta_p * P / self.N + 
                   self.rho * Qt / self.N - 
                   (self.mu + self.gamma_t + self.gamma_p))
        J[2, 3] = self.rho * V / self.N
        
        # Cuarta fila: dQt/dt
        J[3, 2] = self.gamma_t - self.rho * Qt / self.N
        J[3, 3] = -self.rho * V / self.N - self.mu
        
        # Quinta fila: dQp/dt
        J[4, 2] = self.gamma_p
        J[4, 4] = -self.mu
        
        return J
    
    def analizar_estabilidad(self, equilibrio):
        """
        Analiza la estabilidad de un equilibrio mediante autovalores.
        
        Parameters:
        -----------
        equilibrio : tuple
            (S, P, V, Qt, Qp) valores del equilibrio
            
        Returns:
        --------
        dict: Información sobre estabilidad
        """
        J = self.matriz_jacobiana(equilibrio)
        autovalores = np.linalg.eigvals(J)
        
        partes_reales = autovalores.real
        partes_imaginarias = autovalores.imag
        max_parte_real = np.max(partes_reales)
        
        if max_parte_real < -1e-10:
            estabilidad = "Localmente asintóticamente estable"
        elif max_parte_real > 1e-10:
            estabilidad = "Inestable"
        else:
            estabilidad = "No hiperbólico (requiere análisis de orden superior)"
        
        return {
            'autovalores': autovalores,
            'partes_reales': partes_reales,
            'partes_imaginarias': partes_imaginarias,
            'max_parte_real': max_parte_real,
            'estabilidad': estabilidad
        }
    
    def numero_reproductivo_efectivo(self, v):
        """
        Número reproductivo efectivo R(v) evaluado a prevalencia fraccionaria v = V/N.

        Descompone R(v) en la vía de transmisión social y la vía de recaída
        (número reproductivo de recaída). Se cumple R(0) = R_0, y los equilibrios
        endémicos son exactamente las soluciones de R(v) = 1 en (0, 1).

        Parámetros
        ----------
        v : float
            Prevalencia fraccionaria de vapeo, v = V/N, con 0 <= v <= 1.

        Retorna
        -------
        tuple (R_total, R_transmision, R_recaida)
        """
        B = self.phi * self.beta
        Bp = self.phi_p * self.beta_p
        Gamma = self.mu + self.gamma_t + self.gamma_p

        R_trans = (B * self.q * self.mu / (B * v + self.mu)
                   + Bp * (1 - self.q) * self.mu / (Bp * v + self.mu)) / Gamma
        R_recaida = (self.rho * self.gamma_t * v / (self.rho * v + self.mu)) / Gamma

        return R_trans + R_recaida, R_trans, R_recaida

    def criterio_bifurcacion_backward(self):
        """
        Evalúa la condición de bifurcación hacia atrás en R_0 = 1.

        La bifurcación es hacia atrás cuando
            rho * gamma_t > (phi*beta)^2 * q + (phi_p*beta_p)^2 * (1-q),
        condición equivalente a a_1 < 0 sobre la superficie R_0 = 1.

        Retorna
        -------
        dict con los dos miembros de la desigualdad, el veredicto, el valor de
        gamma_t que produce R_0 = 1 y el umbral rho* evaluado en ese punto.
        """
        B = self.phi * self.beta
        Bp = self.phi_p * self.beta_p

        lhs = self.rho * self.gamma_t
        rhs = B**2 * self.q + Bp**2 * (1 - self.q)

        gamma_t_critico = (B * self.q + Bp * (1 - self.q)) - self.mu - self.gamma_p
        rho_critico = rhs / gamma_t_critico if gamma_t_critico > 0 else np.inf

        return {
            'lhs': lhs,
            'rhs': rhs,
            'bifurcacion_backward': lhs > rhs,
            'gamma_t_critico': gamma_t_critico,
            'rho_critico': rho_critico
        }

    def contar_equilibrios(self):
        """
        Cuenta y clasifica los equilibrios endémicos biológicamente admisibles.

        Filtra las raíces de la cúbica al intervalo (0, N] y determina la
        estabilidad local de cada una mediante los autovalores de la jacobiana.

        Retorna
        -------
        dict con el número de equilibrios, el régimen del sistema
        ('supercritico', 'biestable', 'subcritico_sin_endemicos') y la lista
        de equilibrios con su prevalencia, autovalor dominante y estabilidad.
        """
        coefs = self.coeficientes_cubica()
        raices = np.roots(list(coefs))
        reales = [r.real for r in raices
                  if abs(r.imag) < 1e-8 and 1e-10 < r.real <= self.N]

        equilibrios = []
        for V in sorted(reales):
            eq = self.calcular_equilibrio_completo(V)
            eq_tuple = (eq['S'], eq['P'], eq['V'], eq['Qt'], eq['Qp'])
            autovalores = np.linalg.eigvals(self.matriz_jacobiana(eq_tuple))
            lam_max = max(autovalores.real)
            equilibrios.append({
                'V': V,
                'prevalencia': 100 * V / self.N,
                'lambda_max': lam_max,
                'estable': lam_max < 0
            })

        if self.calcular_R0() > 1:
            regimen = 'supercritico'
        elif len(equilibrios) >= 2:
            regimen = 'biestable'
        else:
            regimen = 'subcritico_sin_endemicos'

        return {'n_equilibrios': len(equilibrios),
                'regimen': regimen,
                'equilibrios': equilibrios}

    def sistema_EDO(self, y, t):
        """
        Sistema de ecuaciones diferenciales ordinarias del modelo.
        
        Parameters:
        -----------
        y : array
            [S, P, V, Qt, Qp] valores actuales
        t : float
            Tiempo
            
        Returns:
        --------
        array: Derivadas [dS/dt, dP/dt, dV/dt, dQt/dt, dQp/dt]
        """
        S, P, V, Qt, Qp = y
        
        dS_dt = self.q * self.mu * self.N - self.phi * self.beta * S * V / self.N - self.mu * S
        
        dP_dt = (1 - self.q) * self.mu * self.N - self.phi_p * self.beta_p * P * V / self.N - self.mu * P
        
        dV_dt = (self.phi * self.beta * S * V / self.N + 
                 self.phi_p * self.beta_p * P * V / self.N + 
                 self.rho * Qt * V / self.N - 
                 (self.mu + self.gamma_t + self.gamma_p) * V)
        
        dQt_dt = self.gamma_t * V - self.rho * Qt * V / self.N - self.mu * Qt
        
        dQp_dt = self.gamma_p * V - self.mu * Qp
        
        return [dS_dt, dP_dt, dV_dt, dQt_dt, dQp_dt]
    
    def simular(self, y0, t_span):
        """
        Simula la dinámica temporal del sistema.
        
        Parameters:
        -----------
        y0 : array
            Condiciones iniciales [S0, P0, V0, Qt0, Qp0]
        t_span : array
            Vector de tiempos para la simulación
            
        Returns:
        --------
        numpy.ndarray: Matriz con la evolución temporal de cada compartimento
        """
        solucion = odeint(self.sistema_EDO, y0, t_span)
        return solucion
    
    def calcular_prevalencia(self, solucion):
        """
        Calcula la prevalencia de vapeo en función del tiempo.
        
        Parameters:
        -----------
        solucion : numpy.ndarray
            Matriz de solución temporal del sistema
            
        Returns:
        --------
        numpy.ndarray: Prevalencia porcentual de vapeadores
        """
        V = solucion[:, 2]
        prevalencia = (V / self.N) * 100
        return prevalencia


# ============================================================================
# FUNCIÓN AUXILIAR: GENERAR CONDICIONES INICIALES CONSISTENTES
# ============================================================================

def tiempo_hasta_umbral(t, prevalencia, umbral=1.0):
    """
    Calcula el tiempo en que la prevalencia cruza por debajo de un umbral,
    mediante interpolación lineal entre los nodos de la malla temporal.

    De este modo la precisión del tiempo reportado no queda limitada por el
    paso de la malla (por ejemplo, ~0.1 años con 1000 puntos en [0, 100]).

    Parameters
    ----------
    t : array
        Vector de tiempos de la simulación.
    prevalencia : array
        Prevalencia (%) correspondiente a cada tiempo.
    umbral : float
        Umbral de prevalencia (%) cuyo cruce descendente se busca.

    Returns
    -------
    float o None
        Tiempo interpolado del primer cruce descendente, o None si la
        prevalencia nunca cae por debajo del umbral en el horizonte simulado.
    """
    indices = np.where(prevalencia < umbral)[0]
    if len(indices) == 0:
        return None
    i = indices[0]
    if i == 0:
        return t[0]
    # Interpolación lineal entre (t[i-1], p[i-1]) y (t[i], p[i])
    p0, p1 = prevalencia[i - 1], prevalencia[i]
    return t[i - 1] + (p0 - umbral) / (p0 - p1) * (t[i] - t[i - 1])


def generar_condicion_inicial(N, q, prevalencia_V, Qt0=0, Qp0=0):
    """
    Genera condiciones iniciales que satisfacen S+P+V+Qt+Qp = N.
    
    Distribuye la población restante (N - V0 - Qt0 - Qp0) entre S y P
    según la proporción q.
    
    Parameters:
    -----------
    N : int
        Población total
    q : float
        Proporción de susceptibles
    prevalencia_V : float
        Prevalencia inicial de vapeadores (como fracción, e.g. 0.13 para 13%)
    Qt0 : float
        Población inicial en abandono temporal
    Qp0 : float
        Población inicial en abandono permanente
        
    Returns:
    --------
    list: [S0, P0, V0, Qt0, Qp0] con suma = N
    """
    V0 = prevalencia_V * N
    resto = N - V0 - Qt0 - Qp0
    
    if resto < 0:
        raise ValueError(f"Las condiciones iniciales exceden N: V0={V0}, Qt0={Qt0}, Qp0={Qp0}, suma={V0+Qt0+Qp0} > N={N}")
    
    S0 = q * resto
    P0 = (1 - q) * resto
    
    # Verificación
    suma = S0 + P0 + V0 + Qt0 + Qp0
    assert abs(suma - N) < 1e-6, f"Error en conservación: suma={suma}, N={N}"
    
    return [S0, P0, V0, Qt0, Qp0]


# ============================================================================
# FUNCIONES PARA ANÁLISIS DE SENSIBILIDAD (SECCIÓN 5.9)
# ============================================================================

def analisis_sensibilidad_univariado(params_base, nombre_param, rango_valores, nombre_escenario="Base"):
    """
    Análisis de sensibilidad univariado: varía un parámetro a la vez.
    """
    R0_valores = []
    
    for valor in rango_valores:
        params_temp = params_base.copy()
        params_temp[nombre_param] = valor
        modelo_temp = ModeloVapeo(params_temp)
        R0_valores.append(modelo_temp.calcular_R0())
    
    return {
        'parametro': nombre_param,
        'valores': rango_valores,
        'R0': np.array(R0_valores),
        'escenario': nombre_escenario
    }


def graficar_sensibilidad_univariada(resultados_lista, guardar=True):
    """
    Genera gráficos de sensibilidad univariada para múltiples parámetros.
    """
    etiquetas = {
        'beta': r'$\beta$ (tasa transmisión susceptibles)',
        'beta_p': r'$\beta_p$ (tasa transmisión predispuestos)',
        'gamma_t': r'$\gamma_t$ (tasa abandono temporal)',
        'gamma_p': r'$\gamma_p$ (tasa abandono permanente)',
        'rho': r'$\rho$ (tasa recaída)',
        'phi': r'$\phi$ (factor reducción susceptibles)',
        'phi_p': r'$\phi_p$ (factor reducción predispuestos)',
        'q': r'$q$ (proporción susceptibles)'
    }
    
    n_params = len(resultados_lista)
    n_cols = 2
    n_rows = (n_params + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))
    axes = axes.flatten()
    
    for idx, resultado in enumerate(resultados_lista):
        ax = axes[idx]
        param = resultado['parametro']
        valores = resultado['valores']
        R0 = resultado['R0']
        
        ax.plot(valores, R0, 'b-', linewidth=2)
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label=r'$R_0 = 1$ (umbral crítico)')
        ax.fill_between(valores, 0, 1, alpha=0.1, color='green', label=r'Subcrítico ($R_0 < 1$)')
        ax.fill_between(valores, 1, ax.get_ylim()[1], alpha=0.1, color='red', label=r'Supercrítico ($R_0 > 1$)')
        
        ax.set_xlabel(etiquetas.get(param, param), fontsize=11)
        ax.set_ylabel(r'$R_0$', fontsize=11)
        ax.set_title(f'Sensibilidad de $R_0$ a {etiquetas.get(param, param)}', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    for idx in range(len(resultados_lista), len(axes)):
        axes[idx].axis('off')
    
    fig.subplots_adjust(hspace=0.55, wspace=0.3)
    
    if guardar:
        plt.savefig('sensibilidad_univariada_completa.png', dpi=300, bbox_inches='tight')
        print("Gráfico guardado: sensibilidad_univariada_completa.png")
    
    plt.show()
    plt.close(fig)
    
    return fig


def mapa_calor_bivariado(params_base, param1, rango1, param2, rango2, nombre_escenario="Base"):
    """
    Genera mapa de calor bivariado mostrando R0 en función de dos parámetros.
    """
    n1, n2 = len(rango1), len(rango2)
    R0_matriz = np.zeros((n2, n1))
    
    for i, val1 in enumerate(rango1):
        for j, val2 in enumerate(rango2):
            params_temp = params_base.copy()
            params_temp[param1] = val1
            params_temp[param2] = val2
            modelo_temp = ModeloVapeo(params_temp)
            R0_matriz[j, i] = modelo_temp.calcular_R0()
    
    return {
        'param1': param1,
        'param2': param2,
        'rango1': rango1,
        'rango2': rango2,
        'R0_matriz': R0_matriz,
        'escenario': nombre_escenario
    }


def graficar_mapa_calor(resultado, guardar=True):
    """
    Genera gráfico de mapa de calor bivariado.
    """
    etiquetas = {
        'beta': r'$\beta$',
        'beta_p': r'$\beta_p$',
        'gamma_t': r'$\gamma_t$',
        'gamma_p': r'$\gamma_p$',
        'rho': r'$\rho$',
        'phi': r'$\phi$',
        'phi_p': r'$\phi_p$',
        'q': r'$q$'
    }
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colores = ['darkgreen', 'green', 'yellow', 'orange', 'red', 'darkred']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom', colores, N=n_bins)
    
    im = ax.contourf(resultado['rango1'], resultado['rango2'], resultado['R0_matriz'], 
                     levels=20, cmap=cmap)
    
    contorno = ax.contour(resultado['rango1'], resultado['rango2'], resultado['R0_matriz'],
                          levels=[1.0], colors='white', linewidths=3, linestyles='--')
    ax.clabel(contorno, inline=True, fontsize=10, fmt=r'$R_0=1$')
    
    ax.set_xlabel(etiquetas.get(resultado['param1'], resultado['param1']), fontsize=13)
    ax.set_ylabel(etiquetas.get(resultado['param2'], resultado['param2']), fontsize=13)
    ax.set_title(f"Mapa de calor: $R_0$ en función de {etiquetas.get(resultado['param1'], resultado['param1'])} y {etiquetas.get(resultado['param2'], resultado['param2'])}", 
                 fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$R_0$', fontsize=12)
    
    plt.tight_layout()
    
    if guardar:
        nombre_archivo = f"mapa_calor_{resultado['param1']}_vs_{resultado['param2']}.png"
        plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado: {nombre_archivo}")
    
    plt.show()
    plt.close(fig)
    
    return fig


def encontrar_umbral_critico(params_base, nombre_param, rango_busqueda):
    """
    Encuentra el valor del parámetro donde R0 = 1 (umbral crítico).
    """
    def funcion_R0_menos_1(valor):
        params_temp = params_base.copy()
        params_temp[nombre_param] = valor
        modelo_temp = ModeloVapeo(params_temp)
        return modelo_temp.calcular_R0() - 1.0
    
    try:
        umbral = brentq(funcion_R0_menos_1, rango_busqueda[0], rango_busqueda[1])
        
        params_verificacion = params_base.copy()
        params_verificacion[nombre_param] = umbral
        modelo_verificacion = ModeloVapeo(params_verificacion)
        R0_verificacion = modelo_verificacion.calcular_R0()
        
        return {
            'parametro': nombre_param,
            'valor_umbral': umbral,
            'R0_verificacion': R0_verificacion,
            'encontrado': True
        }
    except ValueError:
        return {
            'parametro': nombre_param,
            'valor_umbral': None,
            'R0_verificacion': None,
            'encontrado': False,
            'mensaje': 'No se encontró umbral en el rango especificado'
        }


def evaluar_intervencion(params_base, params_intervencion, nombre_intervencion, t_max=100):
    """
    Evalúa el impacto de una intervención comparando con escenario base.
    """
    modelo_base = ModeloVapeo(params_base)
    modelo_intervencion = ModeloVapeo(params_intervencion)
    
    R0_base = modelo_base.calcular_R0()
    R0_intervencion = modelo_intervencion.calcular_R0()
    reduccion_R0 = ((R0_base - R0_intervencion) / R0_base) * 100
    
    N = params_base['N']
    q = params_base['q']
    
    # Condiciones iniciales consistentes (prevalencia 13%)
    y0 = generar_condicion_inicial(N, q, 0.13)
    t = np.linspace(0, t_max, 10001)
    
    sol_base = modelo_base.simular(y0, t)
    sol_intervencion = modelo_intervencion.simular(y0, t)
    
    prev_base = modelo_base.calcular_prevalencia(sol_base)
    prev_intervencion = modelo_intervencion.calcular_prevalencia(sol_intervencion)
    
    # Encontrar tiempos hasta prevalencia < 1% (interpolación lineal del cruce)
    tiempo_1pct_base = tiempo_hasta_umbral(t, prev_base, umbral=1.0)
    tiempo_1pct_intervencion = tiempo_hasta_umbral(t, prev_intervencion, umbral=1.0)
    if tiempo_1pct_base is not None and tiempo_1pct_intervencion is not None:
        print(f"  Tiempo hasta <1%: base {tiempo_1pct_base:.2f} años, "
              f"intervención {tiempo_1pct_intervencion:.2f} años")
    
    # Graficar comparación
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(t, prev_base, 'b-', linewidth=2, label='Base')
    axes[0].plot(t, prev_intervencion, 'r-', linewidth=2, label=nombre_intervencion)
    axes[0].axhline(y=1.0, color='black', linestyle=':', alpha=0.5)
    axes[0].set_xlabel('Tiempo (años)', fontsize=12)
    axes[0].set_ylabel('Prevalencia de vapeo (%)', fontsize=12)
    axes[0].set_title(f'Impacto de intervención: {nombre_intervencion}', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    metricas = ['R₀', 'Tiempo hasta\n<1% (años)']
    valores_base = [R0_base, tiempo_1pct_base if tiempo_1pct_base else t_max]
    valores_intervencion = [R0_intervencion, tiempo_1pct_intervencion if tiempo_1pct_intervencion else t_max]
    
    x = np.arange(len(metricas))
    ancho = 0.35
    
    axes[1].bar(x - ancho/2, valores_base, ancho, label='Base', color='blue', alpha=0.7)
    axes[1].bar(x + ancho/2, valores_intervencion, ancho, label=nombre_intervencion, color='red', alpha=0.7)
    
    axes[1].set_ylabel('Valor', fontsize=12)
    axes[1].set_title('Comparación de métricas clave', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metricas, fontsize=11)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'intervencion_{nombre_archivo_seguro(nombre_intervencion)}.png', dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado: intervencion_{nombre_intervencion.lower().replace(' ', '_')}.png")
    plt.show()
    plt.close(fig)
    
    return {
        'nombre': nombre_intervencion,
        'R0_base': R0_base,
        'R0_intervencion': R0_intervencion,
        'reduccion_R0_pct': reduccion_R0,
        'tiempo_1pct_base': tiempo_1pct_base,
        'tiempo_1pct_intervencion': tiempo_1pct_intervencion,
        'solucion_base': sol_base,
        'solucion_intervencion': sol_intervencion,
        'prevalencia_base': prev_base,
        'prevalencia_intervencion': prev_intervencion,
        't': t
    }


def escenario_ajustado_CR(params_ajustados, nombre_escenario="Ajustado CR"):
    """
    Analiza un escenario ajustado para producir R0 > 1 consistente con datos de Costa Rica.
    """
    print(f"\n{'='*80}")
    print(f"ESCENARIO {nombre_escenario.upper()}: ANÁLISIS COMPLETO")
    print(f"{'='*80}\n")
    
    modelo = ModeloVapeo(params_ajustados)
    
    R0 = modelo.calcular_R0()
    print(f"R₀ = {R0:.4f}")
    print(f"Régimen: {'Supercrítico (R₀ > 1)' if R0 > 1 else 'Subcrítico (R₀ < 1)'}")
    
    equilibrios_V = modelo.resolver_equilibrios_endemicos()
    print(f"\nEquilibrios endémicos encontrados: {len(equilibrios_V)}")
    
    equilibrios_completos = []
    for i, V_star in enumerate(equilibrios_V, 1):
        eq = modelo.calcular_equilibrio_completo(V_star)
        print(f"\n  Equilibrio #{i}:")
        print(f"    V* = {V_star:.2f} ({V_star/params_ajustados['N']*100:.2f}% prevalencia)")
        print(f"    S* = {eq['S']:.2f}")
        print(f"    P* = {eq['P']:.2f}")
        print(f"    Qt* = {eq['Qt']:.2f}")
        print(f"    Qp* = {eq['Qp']:.2f}")
        
        eq_tuple = (eq['S'], eq['P'], V_star, eq['Qt'], eq['Qp'])
        estabilidad = modelo.analizar_estabilidad(eq_tuple)
        print(f"    Estabilidad: {estabilidad['estabilidad']}")
        
        equilibrios_completos.append({
            'V': V_star,
            'equilibrio': eq,
            'estabilidad': estabilidad
        })
    
    N = params_ajustados['N']
    q = params_ajustados['q']
    
    # Condiciones iniciales consistentes (prevalencia 13%)
    y0 = generar_condicion_inicial(N, q, 0.13)
    t = np.linspace(0, 100, 10001)
    
    sol = modelo.simular(y0, t)
    prev = modelo.calcular_prevalencia(sol)
    
    print(f"\n{'='*80}")
    print("SIMULACIÓN DINÁMICA (prevalencia inicial 13%)")
    print(f"{'='*80}")
    print(f"Condición inicial: S={y0[0]:.0f}, P={y0[1]:.0f}, V={y0[2]:.0f}, Qt={y0[3]:.0f}, Qp={y0[4]:.0f}")
    print(f"Suma = {sum(y0):.0f} (N = {N})")
    for t_check in [0, 10, 25, 50, 100]:
        idx = int(np.argmin(np.abs(t - t_check)))
        print(f"Prevalencia en t={t_check}: {prev[idx]:.2f}%")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(t, prev, 'r-', linewidth=2)
    axes[0, 0].axhline(y=13, color='green', linestyle='--', alpha=0.5, label='Prevalencia inicial (CR 2025)')
    if len(equilibrios_V) > 0:
        axes[0, 0].axhline(y=equilibrios_V[0]/N*100, color='blue', linestyle=':', 
                          alpha=0.7, label=f'Equilibrio endémico ({equilibrios_V[0]/N*100:.1f}%)')
    axes[0, 0].set_xlabel('Tiempo (años)', fontsize=12)
    axes[0, 0].set_ylabel('Prevalencia de vapeo (%)', fontsize=12)
    axes[0, 0].set_title('Evolución de prevalencia', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(t, sol[:, 2], 'r-', linewidth=2)
    if len(equilibrios_V) > 0:
        axes[0, 1].axhline(y=equilibrios_V[0], color='blue', linestyle=':', 
                          alpha=0.7, label=f'V* = {equilibrios_V[0]:.0f}')
    axes[0, 1].set_xlabel('Tiempo (años)', fontsize=12)
    axes[0, 1].set_ylabel('Vapeadores (V)', fontsize=12)
    axes[0, 1].set_title('Población vapeadora', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(t, sol[:, 0], 'b-', linewidth=2, label='Susceptibles (S)')
    axes[1, 0].plot(t, sol[:, 1], 'g-', linewidth=2, label='Predispuestos (P)')
    axes[1, 0].set_xlabel('Tiempo (años)', fontsize=12)
    axes[1, 0].set_ylabel('Población', fontsize=12)
    axes[1, 0].set_title('Susceptibles y predispuestos', fontsize=13, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(t, sol[:, 3], 'orange', linewidth=2, label='Abandono temporal (Qt)')
    axes[1, 1].plot(t, sol[:, 4], 'purple', linewidth=2, label='Abandono permanente (Qp)')
    axes[1, 1].set_xlabel('Tiempo (años)', fontsize=12)
    axes[1, 1].set_ylabel('Población', fontsize=12)
    axes[1, 1].set_title('Abandono temporal y permanente', fontsize=13, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'escenario_{nombre_archivo_seguro(nombre_escenario)}.png', dpi=300, bbox_inches='tight')
    print(f"\nGráfico guardado: escenario_{nombre_escenario.lower().replace(' ', '_')}.png")
    plt.show()
    plt.close(fig)
    
    return {
        'nombre': nombre_escenario,
        'R0': R0,
        'equilibrios': equilibrios_completos,
        'solucion': sol,
        'prevalencia': prev,
        't': t,
        'modelo': modelo
    }


def analisis_completo_escenario(params, nombre_escenario):
    """
    Realiza el análisis completo para un escenario paramétrico.
    """
    print(f"\n{'='*80}")
    print(f"ANÁLISIS DEL ESCENARIO: {nombre_escenario}")
    print(f"{'='*80}\n")
    
    modelo = ModeloVapeo(params)
    resultados = {}
    
    # 1. Número reproductivo básico
    R0 = modelo.calcular_R0()
    print(f"1. NÚMERO REPRODUCTIVO BÁSICO")
    print(f"   R₀ = {R0:.4f}")
    print(f"   Régimen: {'Supercrítico (R₀ > 1)' if R0 > 1 else 'Subcrítico (R₀ < 1)' if R0 < 1 else 'Crítico (R₀ = 1)'}")
    resultados['R0'] = R0
    
    # 2. Equilibrio libre de vapeo
    eq_libre = modelo.equilibrio_libre_vapeo()
    print(f"\n2. EQUILIBRIO LIBRE DE VAPEO")
    print(f"   (S*, P*, V*, Qt*, Qp*) = ({eq_libre[0]:.1f}, {eq_libre[1]:.1f}, {eq_libre[2]:.1f}, {eq_libre[3]:.1f}, {eq_libre[4]:.1f})")
    resultados['equilibrio_libre'] = eq_libre
    
    # Estabilidad del equilibrio libre
    estabilidad_libre = modelo.analizar_estabilidad(eq_libre)
    print(f"   Estabilidad: {estabilidad_libre['estabilidad']}")
    print(f"   Autovalores: {estabilidad_libre['autovalores']}")
    resultados['estabilidad_libre'] = estabilidad_libre
    
    # 3. Coeficientes de la ecuación cúbica
    a_3, a_2, a_1, a_0 = modelo.coeficientes_cubica()
    print(f"\n3. COEFICIENTES DE LA ECUACIÓN CÚBICA")
    print(f"   a₃ = {a_3:.6e}")
    print(f"   a₂ = {a_2:.6e}")
    print(f"   a₁ = {a_1:.6e}")
    print(f"   a₀ = {a_0:.6e}")
    resultados['coeficientes'] = (a_3, a_2, a_1, a_0)
    
    # 4. Discriminante cúbico
    Delta = modelo.discriminante_cubica()
    print(f"\n4. DISCRIMINANTE CÚBICO")
    print(f"   Δ = {Delta:.6e}")
    if Delta > 0:
        print(f"   Interpretación: 3 raíces reales distintas")
    elif Delta == 0:
        print(f"   Interpretación: Al menos una raíz múltiple")
    else:
        print(f"   Interpretación: 1 raíz real, 2 raíces complejas conjugadas")
    resultados['discriminante'] = Delta
    
    # 5. Puntos críticos
    Delta_prima, puntos_crit = modelo.puntos_criticos_cubica()
    print(f"\n5. PUNTOS CRÍTICOS DE f(V*)")
    print(f"   Δ' = {Delta_prima:.6e}")
    if len(puntos_crit) > 0:
        print(f"   Puntos críticos: {[f'{p:.2f}' for p in puntos_crit]}")
    else:
        print(f"   No hay puntos críticos reales")
    resultados['puntos_criticos'] = puntos_crit
    
    # 6. Equilibrios endémicos
    equilibrios_V = modelo.resolver_equilibrios_endemicos()
    print(f"\n6. EQUILIBRIOS ENDÉMICOS")
    print(f"   Número de equilibrios endémicos (V* > 0): {len(equilibrios_V)}")
    
    equilibrios_completos = []
    for i, V_star in enumerate(equilibrios_V, 1):
        eq_completo = modelo.calcular_equilibrio_completo(V_star)
        print(f"\n   Equilibrio endémico #{i}:")
        print(f"      V* = {V_star:.2f} vapeadores")
        print(f"      S* = {eq_completo['S']:.2f}")
        print(f"      P* = {eq_completo['P']:.2f}")
        print(f"      Qt* = {eq_completo['Qt']:.2f}")
        print(f"      Qp* = {eq_completo['Qp']:.2f}")
        print(f"      Total = {eq_completo['Total']:.2f} (Error: {eq_completo['Error_conservacion']:.6f})")
        
        equilibrio_tuple = (eq_completo['S'], eq_completo['P'], V_star, 
                           eq_completo['Qt'], eq_completo['Qp'])
        estabilidad = modelo.analizar_estabilidad(equilibrio_tuple)
        print(f"      Estabilidad: {estabilidad['estabilidad']}")
        print(f"      Autovalor dominante (parte real): {estabilidad['max_parte_real']:.6f}")
        
        equilibrios_completos.append({
            'V': V_star,
            'equilibrio': eq_completo,
            'estabilidad': estabilidad
        })
    
    resultados['equilibrios_endemicos'] = equilibrios_completos
    resultados['modelo'] = modelo
    
    return resultados


def simulaciones_dinamicas(modelo, nombre_escenario, condiciones_iniciales_lista, t_max=100):
    """
    Realiza simulaciones dinámicas con diferentes condiciones iniciales.
    """
    print(f"\n{'='*80}")
    print(f"SIMULACIONES DINÁMICAS - {nombre_escenario}")
    print(f"{'='*80}\n")
    
    t = np.linspace(0, t_max, 10001)
    resultados_sim = []
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Simulaciones dinámicas - {nombre_escenario}', fontsize=14, fontweight='bold')
    
    colores = ['blue', 'red', 'green', 'orange', 'purple']
    
    for idx, (y0, descripcion) in enumerate(condiciones_iniciales_lista):
        print(f"Simulación {idx+1}: {descripcion}")
        print(f"   Condición inicial: S={y0[0]:.0f}, P={y0[1]:.0f}, V={y0[2]:.0f}, Qt={y0[3]:.0f}, Qp={y0[4]:.0f}")
        print(f"   Suma = {sum(y0):.0f} (N = {modelo.N})")
        
        sol = modelo.simular(y0, t)
        prevalencia = modelo.calcular_prevalencia(sol)
        
        prevalencia_inicial = (y0[2] / modelo.N) * 100
        prevalencia_final = prevalencia[-1]
        tiempo_hasta_1_porciento = tiempo_hasta_umbral(t, prevalencia, umbral=1.0)
        
        print(f"   Prevalencia inicial: {prevalencia_inicial:.2f}%")
        print(f"   Prevalencia final (t={t_max}): {prevalencia_final:.4f}%")
        if tiempo_hasta_1_porciento:
            print(f"   Tiempo hasta prevalencia < 1%: {tiempo_hasta_1_porciento:.2f} años")
        else:
            print(f"   Prevalencia no alcanza < 1% en {t_max} años")
        print()
        
        color = colores[idx % len(colores)]
        
        axes[0, 0].plot(t, sol[:, 0], color=color, alpha=0.7, label=f'{descripcion} - S')
        axes[0, 0].plot(t, sol[:, 1], color=color, alpha=0.7, linestyle='--', label=f'{descripcion} - P')
        axes[0, 0].set_xlabel('Tiempo (años)')
        axes[0, 0].set_ylabel('Población')
        axes[0, 0].set_title('Susceptibles y predispuestos')
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(t, sol[:, 2], color=color, label=descripcion, linewidth=2)
        axes[0, 1].set_xlabel('Tiempo (años)')
        axes[0, 1].set_ylabel('Vapeadores (V)')
        axes[0, 1].set_title('Evolución de vapeadores activos')
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(t, sol[:, 3], color=color, alpha=0.7, label=f'{descripcion} - Qt')
        axes[1, 0].plot(t, sol[:, 4], color=color, alpha=0.7, linestyle='--', label=f'{descripcion} - Qp')
        axes[1, 0].set_xlabel('Tiempo (años)')
        axes[1, 0].set_ylabel('Población')
        axes[1, 0].set_title('Abandono temporal y permanente')
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(t, prevalencia, color=color, label=descripcion, linewidth=2)
        axes[1, 1].axhline(y=1.0, color='black', linestyle=':', alpha=0.5, label='1% prevalencia')
        axes[1, 1].set_xlabel('Tiempo (años)')
        axes[1, 1].set_ylabel('Prevalencia de vapeo (%)')
        axes[1, 1].set_title('Prevalencia de vapeo en el tiempo')
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)
        
        resultados_sim.append({
            'descripcion': descripcion,
            'y0': y0,
            'solucion': sol,
            'prevalencia': prevalencia,
            'prevalencia_inicial': prevalencia_inicial,
            'prevalencia_final': prevalencia_final,
            'tiempo_hasta_1pct': tiempo_hasta_1_porciento
        })
    
    plt.tight_layout()
    plt.savefig(f'simulaciones_{nombre_archivo_seguro(nombre_escenario)}.png', dpi=300, bbox_inches='tight')
    print(f"\n   Gráfico guardado como: simulaciones_{nombre_escenario.lower().replace(' ', '_')}.png")
    plt.show()
    plt.close(fig)
    
    return resultados_sim


# ============================================================================
# ESCENARIOS PARAMÉTRICOS
# ============================================================================

# Escenario Base (calibrado según literatura)
params_base = {
    'beta': 0.18,
    'beta_p': 0.50,
    'gamma_t': 0.17,
    'gamma_p': 0.13,
    'rho': 0.121,
    'phi': 0.65,
    'phi_p': 0.75,
    'q': 0.62,
    'mu': 0.0125,
    'N': 10000
}

# Escenario Conservador (ajustado por contexto costarricense)
params_conservador = {
    'beta': 0.18,
    'beta_p': 0.27,
    'gamma_t': 0.22,
    'gamma_p': 0.11,
    'rho': 0.121,
    'phi': 0.75,
    'phi_p': 0.82,
    'q': 0.62,
    'mu': 0.0125,
    'N': 10000
}

# ============================================================================
# EJECUCIÓN PRINCIPAL (SECCIONES 5.7 Y 5.8)
# ============================================================================

print("\n" + "="*80)
print("IMPLEMENTACIÓN COMPUTACIONAL DEL MODELO DE VAPEO")
print("="*80)

# Análisis de equilibrios
resultados_base = analisis_completo_escenario(params_base, "BASE")
resultados_conservador = analisis_completo_escenario(params_conservador, "CONSERVADOR")

# Resumen comparativo
print(f"\n{'='*80}")
print("RESUMEN COMPARATIVO DE EQUILIBRIOS")
print(f"{'='*80}\n")
print(f"Escenario base:")
print(f"  R₀ = {resultados_base['R0']:.4f}")
print(f"  Equilibrios endémicos: {len(resultados_base['equilibrios_endemicos'])}")

print(f"\nEscenario conservador:")
print(f"  R₀ = {resultados_conservador['R0']:.4f}")
print(f"  Equilibrios endémicos: {len(resultados_conservador['equilibrios_endemicos'])}")

# Simulaciones dinámicas con condiciones iniciales CONSISTENTES
N = 10000
q = 0.62

condiciones_base = [
    (generar_condicion_inicial(N, q, 0.05), "Prevalencia inicial 5%"),
    (generar_condicion_inicial(N, q, 0.13), "Prevalencia inicial 13% (CR 2025)"),
    (generar_condicion_inicial(N, q, 0.20), "Prevalencia inicial 20%"),
    (generar_condicion_inicial(N, q, 0.10, Qt0=500, Qp0=300), "Con abandono previo"),
]

print("\n" + "="*80)
print("INICIANDO SIMULACIONES DINÁMICAS")
print("="*80)

# Verificar que todas las condiciones iniciales suman N
print("\nVerificación de condiciones iniciales:")
for y0, desc in condiciones_base:
    print(f"  {desc}: S={y0[0]:.0f}, P={y0[1]:.0f}, V={y0[2]:.0f}, Qt={y0[3]:.0f}, Qp={y0[4]:.0f}, Suma={sum(y0):.0f}")

sim_base = simulaciones_dinamicas(
    resultados_base['modelo'], 
    "ESCENARIO BASE", 
    condiciones_base,
    t_max=100
)

# Extraer prevalencias en tiempos específicos para el caso 13% (índice 1)
print("\n--- PREVALENCIAS ESPECÍFICAS ESCENARIO BASE (13%) ---")
t = np.linspace(0, 100, 10001)
prev_13 = sim_base[1]['prevalencia']
for t_check in [10, 25, 50, 100]:
    idx = np.argmin(np.abs(t - t_check))
    print(f"  Prevalencia en t={t_check}: {prev_13[idx]:.2f}%")

sim_conservador = simulaciones_dinamicas(
    resultados_conservador['modelo'],
    "ESCENARIO CONSERVADOR",
    condiciones_base,
    t_max=100
)

# Extraer prevalencias en tiempos específicos para el caso 13% conservador (índice 1)
print("\n--- PREVALENCIAS ESPECÍFICAS ESCENARIO CONSERVADOR (13%) ---")
prev_13_cons = sim_conservador[1]['prevalencia']
for t_check in [10, 25, 50, 100]:
    idx = np.argmin(np.abs(t - t_check))
    print(f"  Prevalencia en t={t_check}: {prev_13_cons[idx]:.2f}%")

# Comparación directa de escenarios con prevalencia 13%
print("\nGenerando gráfico comparativo Base vs Conservador (prevalencia 13%)...")
y0_13 = generar_condicion_inicial(N, q, 0.13)
t = np.linspace(0, 100, 1000)

sol_base_13 = resultados_base['modelo'].simular(y0_13, t)
sol_cons_13 = resultados_conservador['modelo'].simular(y0_13, t)
prev_base_13 = resultados_base['modelo'].calcular_prevalencia(sol_base_13)
prev_cons_13 = resultados_conservador['modelo'].calcular_prevalencia(sol_cons_13)

fig_comp, ax_comp = plt.subplots(figsize=(12, 6))
ax_comp.plot(t, prev_base_13, 'b-', linewidth=2.5, label=f'Escenario base (R₀={resultados_base["R0"]:.2f})')
ax_comp.plot(t, prev_cons_13, 'r-', linewidth=2.5, label=f'Escenario conservador (R₀={resultados_conservador["R0"]:.2f})')
ax_comp.axhline(y=13, color='green', linestyle='--', alpha=0.5, label='Prevalencia inicial CR 2025')
ax_comp.axhline(y=1, color='black', linestyle=':', alpha=0.5, label='Umbral 1%')
ax_comp.set_xlabel('Tiempo (años)', fontsize=13)
ax_comp.set_ylabel('Prevalencia de vapeo (%)', fontsize=13)
ax_comp.set_title('Comparación de escenarios: prevalencia inicial 13%', fontsize=14, fontweight='bold')
ax_comp.legend(fontsize=11)
ax_comp.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('comparacion_escenarios_13pct.png', dpi=300, bbox_inches='tight')
print("Gráfico guardado: comparacion_escenarios_13pct.png")
plt.show()
plt.close(fig_comp)


# ============================================================================
# ANÁLISIS DE SENSIBILIDAD PARAMÉTRICA (SECCIÓN 5.9)
# ============================================================================

print("\n" + "="*80)
print("SECCIÓN 5.9: ANÁLISIS DE SENSIBILIDAD PARAMÉTRICA")
print("="*80)

# 5.9.1: Análisis univariado
print("\n" + "-"*80)
print("5.9.1: ANÁLISIS DE SENSIBILIDAD UNIVARIADO")
print("-"*80)

rangos_params = {
    'beta': np.linspace(0.10, 0.40, 50),
    'beta_p': np.linspace(0.20, 0.80, 50),
    'gamma_t': np.linspace(0.05, 0.30, 50),
    'gamma_p': np.linspace(0.02, 0.20, 50),
    'rho': np.linspace(0.05, 0.30, 50),
    'phi': np.linspace(0.50, 0.95, 50),
    'phi_p': np.linspace(0.60, 0.98, 50),
    'q': np.linspace(0.40, 0.80, 50)
}

resultados_univariados = []
for param, rango in rangos_params.items():
    print(f"\nAnalizando sensibilidad de R₀ a {param}...")
    resultado = analisis_sensibilidad_univariado(params_base, param, rango)
    resultados_univariados.append(resultado)
    
    R0_array = resultado['R0']
    if R0_array.min() < 1.0 < R0_array.max():
        umbral = encontrar_umbral_critico(params_base, param, (rango.min(), rango.max()))
        print(f"  Umbral crítico (R₀=1): {param} = {umbral['valor_umbral']:.4f}")
    else:
        print(f"  No hay umbral crítico en el rango explorado")

print("\nGenerando gráficos de sensibilidad univariada...")
graficar_sensibilidad_univariada(resultados_univariados)

# 5.9.2: Mapas de calor bivariados
print("\n" + "-"*80)
print("5.9.2: ANÁLISIS BIVARIADO - MAPAS DE CALOR")
print("-"*80)

combinaciones_clave = [
    ('beta', 'gamma_t', np.linspace(0.10, 0.40, 40), np.linspace(0.05, 0.30, 40)),
    ('beta_p', 'gamma_p', np.linspace(0.20, 0.80, 40), np.linspace(0.02, 0.20, 40)),
    ('phi', 'phi_p', np.linspace(0.50, 0.95, 40), np.linspace(0.60, 0.98, 40)),
]

for param1, param2, rango1, rango2 in combinaciones_clave:
    print(f"\nGenerando mapa de calor: {param1} vs {param2}...")
    resultado_mapa = mapa_calor_bivariado(params_base, param1, rango1, param2, rango2)
    graficar_mapa_calor(resultado_mapa)

# 5.9.3: Escenarios de intervención
print("\n" + "-"*80)
print("5.9.3: EVALUACIÓN DE INTERVENCIONES")
print("-"*80)

# Intervención 1: Programas de cesación mejorados
params_cesacion = params_base.copy()
params_cesacion['gamma_t'] = 0.25
params_cesacion['gamma_p'] = 0.18
print("\nIntervención 1: Programas de cesación mejorados")
print(f"  γt: {params_base['gamma_t']} → {params_cesacion['gamma_t']}")
print(f"  γp: {params_base['gamma_p']} → {params_cesacion['gamma_p']}")
resultado_cesacion = evaluar_intervencion(params_base, params_cesacion, "Cesación mejorada")

# Intervención 2: Campañas de concientización
params_concientizacion = params_base.copy()
params_concientizacion['beta'] = 0.12
params_concientizacion['beta_p'] = 0.35
print("\nIntervención 2: Campañas de concientización")
print(f"  β: {params_base['beta']} → {params_concientizacion['beta']}")
print(f"  βp: {params_base['beta_p']} → {params_concientizacion['beta_p']}")
resultado_concientizacion = evaluar_intervencion(params_base, params_concientizacion, "Concientizacion")

# Intervención 3: Testimonios de exvapeadores
params_testimonios = params_base.copy()
params_testimonios['phi'] = 0.50
params_testimonios['phi_p'] = 0.60
print("\nIntervención 3: Testimonios de exvapeadores")
print(f"  φ: {params_base['phi']} → {params_testimonios['phi']}")
print(f"  φp: {params_base['phi_p']} → {params_testimonios['phi_p']}")
resultado_testimonios = evaluar_intervencion(params_base, params_testimonios, "Testimonios exvapeadores")

# Intervención 4: Intervención combinada
params_combinada = params_base.copy()
params_combinada['gamma_t'] = 0.25
params_combinada['gamma_p'] = 0.18
params_combinada['beta'] = 0.12
params_combinada['beta_p'] = 0.35
params_combinada['phi'] = 0.50
params_combinada['phi_p'] = 0.60
print("\nIntervención 4: Intervención combinada")
resultado_combinada = evaluar_intervencion(params_base, params_combinada, "Intervencion combinada")

# Resumen comparativo de intervenciones
print("\n" + "="*80)
print("RESUMEN COMPARATIVO DE INTERVENCIONES")
print("="*80)

intervenciones_resumen = [
    ("Base", params_base['beta'], params_base['gamma_t'], resultados_base['R0']),
    ("Cesación mejorada", params_cesacion['beta'], params_cesacion['gamma_t'], resultado_cesacion['R0_intervencion']),
    ("Concientización", params_concientizacion['beta'], params_concientizacion['gamma_t'], resultado_concientizacion['R0_intervencion']),
    ("Testimonios", params_testimonios['beta'], params_testimonios['gamma_t'], resultado_testimonios['R0_intervencion']),
    ("Combinada", params_combinada['beta'], params_combinada['gamma_t'], resultado_combinada['R0_intervencion'])
]

print(f"\n{'Intervención':<25} {'R₀':<10} {'Reducción vs Base':<20} {'Régimen':<15}")
print("-"*80)
for nombre, beta, gamma_t, R0 in intervenciones_resumen:
    reduccion = ((resultados_base['R0'] - R0) / resultados_base['R0']) * 100 if nombre != "Base" else 0
    regimen = "Supercrítico" if R0 > 1 else "Subcrítico"
    print(f"{nombre:<25} {R0:<10.4f} {reduccion:<20.2f}% {regimen:<15}")

# Gráfico comparativo de intervenciones
fig_int, ax_int = plt.subplots(figsize=(12, 6))

nombres = [i[0] for i in intervenciones_resumen]
R0_valores = [i[3] for i in intervenciones_resumen]
colores_barras = ['gray' if nombre == "Base" else 'blue' if R0 < 1 else 'red' 
                  for nombre, R0 in zip(nombres, R0_valores)]

bars = ax_int.bar(nombres, R0_valores, color=colores_barras, alpha=0.7, edgecolor='black', linewidth=1.5)
ax_int.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Umbral crítico (R₀=1)')

for bar, valor in zip(bars, R0_valores):
    height = bar.get_height()
    ax_int.text(bar.get_x() + bar.get_width()/2., height,
            f'{valor:.3f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax_int.set_ylabel('R₀', fontsize=13, fontweight='bold')
ax_int.set_title('Comparación de R₀ entre escenarios de intervención', fontsize=14, fontweight='bold')
ax_int.legend(fontsize=11)
ax_int.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig('comparacion_intervenciones_R0.png', dpi=300, bbox_inches='tight')
print("\nGráfico guardado: comparacion_intervenciones_R0.png")
plt.show()
plt.close(fig_int)

# 5.9.4: Escenario ajustado para Costa Rica (R0 > 1)
print("\n" + "-"*80)
print("5.9.4: ESCENARIO AJUSTADO PARA COSTA RICA (R₀ > 1)")
print("-"*80)

# Escenario 1: Ajuste moderado
params_ajustado_moderado = {
    'beta': 0.28,
    'beta_p': 0.65,
    'gamma_t': 0.10,
    'gamma_p': 0.06,
    'rho': 0.121,
    'phi': 0.85,
    'phi_p': 0.90,
    'q': 0.62,
    'mu': 0.0125,
    'N': 10000
}

print("\nEscenario ajustado moderado:")
print("Justificación de cambios:")
print(f"  β: {params_base['beta']} → {params_ajustado_moderado['beta']} (mayor transmisión, marketing agresivo)")
print(f"  βp: {params_base['beta_p']} → {params_ajustado_moderado['beta_p']} (mayor susceptibilidad)")
print(f"  γt: {params_base['gamma_t']} → {params_ajustado_moderado['gamma_t']} (menor cesación temporal)")
print(f"  γp: {params_base['gamma_p']} → {params_ajustado_moderado['gamma_p']} (menor cesación permanente)")
print(f"  φ: {params_base['phi']} → {params_ajustado_moderado['phi']} (menor influencia protectora)")
print(f"  φp: {params_base['phi_p']} → {params_ajustado_moderado['phi_p']} (normalización social)")

resultado_moderado = escenario_ajustado_CR(params_ajustado_moderado, "Ajustado moderado")

# Escenario 2: Ajuste alto
params_ajustado_alto = {
    'beta': 0.35,
    'beta_p': 0.75,
    'gamma_t': 0.08,
    'gamma_p': 0.04,
    'rho': 0.121,
    'phi': 0.90,
    'phi_p': 0.95,
    'q': 0.62,
    'mu': 0.0125,
    'N': 10000
}

print("\nEscenario ajustado alto:")
resultado_alto = escenario_ajustado_CR(params_ajustado_alto, "Ajustado alto")

# 5.9.5: Comparación de todos los escenarios
print("\n" + "="*80)
print("5.9.5: COMPARACIÓN INTEGRAL DE ESCENARIOS")
print("="*80)

escenarios_comparacion = [
    ("Base", resultados_base['R0'], len(resultados_base['equilibrios_endemicos'])),
    ("Conservador", resultados_conservador['R0'], len(resultados_conservador['equilibrios_endemicos'])),
    ("Ajustado moderado", resultado_moderado['R0'], len(resultado_moderado['equilibrios'])),
    ("Ajustado alto", resultado_alto['R0'], len(resultado_alto['equilibrios']))
]

print(f"\n{'Escenario':<25} {'R₀':<10} {'Equilibrios endémicos':<25} {'Régimen':<15}")
print("-"*80)
for nombre, R0, n_eq in escenarios_comparacion:
    regimen = "Supercrítico" if R0 > 1 else "Subcrítico"
    print(f"{nombre:<25} {R0:<10.4f} {n_eq:<25} {regimen:<15}")

# Gráfico comparativo de prevalencia en el tiempo (todos los escenarios)
print("\nGenerando gráfico comparativo de evolución temporal...")

fig_todos, ax_todos = plt.subplots(figsize=(14, 7))

y0_13 = generar_condicion_inicial(N, q, 0.13)
t = np.linspace(0, 100, 10001)

# Base
sol_base = resultados_base['modelo'].simular(y0_13, t)
prev_base = resultados_base['modelo'].calcular_prevalencia(sol_base)
ax_todos.plot(t, prev_base, 'b-', linewidth=2.5, label=f'Base (R₀={resultados_base["R0"]:.2f})')

# Conservador
sol_cons = resultados_conservador['modelo'].simular(y0_13, t)
prev_cons = resultados_conservador['modelo'].calcular_prevalencia(sol_cons)
ax_todos.plot(t, prev_cons, 'g-', linewidth=2.5, label=f'Conservador (R₀={resultados_conservador["R0"]:.2f})')

# Moderado
prev_mod = resultado_moderado['prevalencia']
ax_todos.plot(t, prev_mod, 'orange', linewidth=2.5, label=f'Ajustado moderado (R₀={resultado_moderado["R0"]:.2f})')

# Alto
prev_alto = resultado_alto['prevalencia']
ax_todos.plot(t, prev_alto, 'r-', linewidth=2.5, label=f'Ajustado alto (R₀={resultado_alto["R0"]:.2f})')

# Líneas de referencia
ax_todos.axhline(y=13, color='purple', linestyle='--', alpha=0.6, linewidth=1.5, label='Prevalencia inicial CR 2025 (13%)')
ax_todos.axhline(y=1, color='black', linestyle=':', alpha=0.5, linewidth=1.5, label='Umbral 1%')

ax_todos.set_xlabel('Tiempo (años)', fontsize=13)
ax_todos.set_ylabel('Prevalencia de vapeo (%)', fontsize=13)
ax_todos.set_title('Comparación de escenarios: evolución temporal desde prevalencia inicial 13%', 
             fontsize=14, fontweight='bold')
ax_todos.legend(fontsize=11, loc='best')
ax_todos.grid(True, alpha=0.3)
ax_todos.set_xlim([0, 100])

plt.tight_layout()
plt.savefig('comparacion_todos_escenarios_temporal.png', dpi=300, bbox_inches='tight')
print("Gráfico guardado: comparacion_todos_escenarios_temporal.png")
plt.show()
plt.close(fig_todos)

# Resumen de equilibrios endémicos en escenarios supercríticos
print("\n" + "="*80)
print("EQUILIBRIOS ENDÉMICOS EN ESCENARIOS SUPERCRÍTICOS")
print("="*80)

if len(resultado_moderado['equilibrios']) > 0:
    print(f"\nEscenario ajustado moderado (R₀={resultado_moderado['R0']:.4f}):")
    for i, eq in enumerate(resultado_moderado['equilibrios'], 1):
        prev_eq = (eq['V'] / N) * 100
        print(f"  Equilibrio #{i}: V* = {eq['V']:.2f} ({prev_eq:.2f}% prevalencia)")
        print(f"    Estabilidad: {eq['estabilidad']['estabilidad']}")

if len(resultado_alto['equilibrios']) > 0:
    print(f"\nEscenario ajustado alto (R₀={resultado_alto['R0']:.4f}):")
    for i, eq in enumerate(resultado_alto['equilibrios'], 1):
        prev_eq = (eq['V'] / N) * 100
        print(f"  Equilibrio #{i}: V* = {eq['V']:.2f} ({prev_eq:.2f}% prevalencia)")
        print(f"    Estabilidad: {eq['estabilidad']['estabilidad']}")

# 5.9.6: Implicaciones para políticas públicas
print("\n" + "="*80)
print("5.9.6: IMPLICACIONES PARA POLÍTICAS PÚBLICAS")
print("="*80)

print("\nPRINCIPALES HALLAZGOS DEL ANÁLISIS DE SENSIBILIDAD:")
print("\n1. Parámetros más influyentes sobre R₀:")
print("   - Tasas de transmisión (β, βp): Impacto directo y alto")
print("   - Tasas de cesación (γt, γp): Impacto inverso significativo")
print("   - Factores de influencia protectora (φ, φp): Impacto moderado")

print("\n2. Umbrales críticos identificados:")
for resultado in resultados_univariados:
    R0_array = resultado['R0']
    if R0_array.min() < 1.0 < R0_array.max():
        param = resultado['parametro']
        umbral = encontrar_umbral_critico(params_base, param, 
                                         (resultado['valores'].min(), resultado['valores'].max()))
        if umbral['encontrado']:
            print(f"   - {param}: valor crítico ≈ {umbral['valor_umbral']:.4f}")

print("\n3. Efectividad relativa de intervenciones (reducción de R₀):")
resultados_intervenciones = [
    ("Cesación mejorada", resultado_cesacion['reduccion_R0_pct']),
    ("Concientización", resultado_concientizacion['reduccion_R0_pct']),
    ("Testimonios exvapeadores", resultado_testimonios['reduccion_R0_pct']),
    ("Intervención combinada", resultado_combinada['reduccion_R0_pct'])
]
for nombre, reduccion in sorted(resultados_intervenciones, key=lambda x: x[1], reverse=True):
    print(f"   - {nombre}: {reduccion:.2f}% de reducción")

# ============================================================================
# ANÁLISIS DE MULTIPLICIDAD DE EQUILIBRIOS Y BIFURCACIÓN HACIA ATRÁS
# ============================================================================

print("\n" + "=" * 80)
print("5.9.10: ANÁLISIS DE MULTIPLICIDAD DE EQUILIBRIOS")
print("=" * 80)

# --- Criterio de bifurcación hacia atrás para el escenario base ---
modelo_base = ModeloVapeo(params_base)
crit = modelo_base.criterio_bifurcacion_backward()
print(f"\nEscenario base:")
print(f"  rho * gamma_t          = {crit['lhs']:.5f}")
print(f"  (phi*beta)^2 q + ...   = {crit['rhs']:.5f}")
print(f"  Bifurcacion hacia atras: {crit['bifurcacion_backward']}")
print(f"  gamma_t que da R0 = 1  = {crit['gamma_t_critico']:.5f}")
print(f"  Umbral rho*            = {crit['rho_critico']:.4f}")

# --- Configuración biestable de referencia ---
params_biestable = {'beta': 0.22, 'beta_p': 0.50, 'gamma_t': 0.20, 'gamma_p': 0.09,
                    'rho': 2.50, 'phi': 0.70, 'phi_p': 0.80, 'q': 0.58,
                    'mu': 0.0125, 'N': 10000}
modelo_bi = ModeloVapeo(params_biestable)
res = modelo_bi.contar_equilibrios()
print(f"\nConfiguracion biestable: R0 = {modelo_bi.calcular_R0():.4f}, "
      f"regimen = {res['regimen']}")
for i, eq in enumerate(res['equilibrios'], 1):
    print(f"  Equilibrio #{i}: V* = {eq['V']:.2f} ({eq['prevalencia']:.2f}%), "
          f"lambda_max = {eq['lambda_max']:+.4f}, "
          f"{'estable' if eq['estable'] else 'inestable'}")

# --- Figura: mapa del número de equilibrios en (rho, gamma_t) ---
print("\nGenerando mapa del numero de equilibrios...")
rhos = np.linspace(0.05, 3.10, 300)
gts = np.linspace(0.02, 0.30, 300)
M = np.zeros((len(gts), len(rhos)))
for i, r in enumerate(rhos):
    for j, g in enumerate(gts):
        p = dict(params_base, rho=r, gamma_t=g)
        m = ModeloVapeo(p)
        res_ij = m.contar_equilibrios()
        M[j, i] = (2 if res_ij['regimen'] == 'supercritico'
                   else (1 if res_ij['regimen'] == 'biestable' else 0))

from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(9, 6))
ax.pcolormesh(rhos, gts, M, cmap=ListedColormap(['#1a7d3a', '#c0392b', '#e8a33d']),
              shading='auto')
gt_um = modelo_base.criterio_bifurcacion_backward()['gamma_t_critico']
ax.axhline(gt_um, color='white', ls='--', lw=1.6)
ax.text(1.75, gt_um + 0.006, r'$R_0=1$', color='white', fontsize=11, ha='center')
ax.plot(params_base['rho'], params_base['gamma_t'], marker='*', ms=20,
        color='white', mec='black', mew=1.2, ls='', zorder=5)
ax.plot(0.93, params_base['gamma_t'], marker='o', ms=10,
        color='white', mec='black', mew=1.2, ls='', zorder=5)
ax.set_xlabel(r'$\rho$ (tasa de recaída)', fontsize=12)
ax.set_ylabel(r'$\gamma_t$ (tasa de abandono temporal)', fontsize=12)
ax.set_title('Número de equilibrios endémicos en el espacio ' + r'$(\rho,\gamma_t)$',
             fontsize=13, fontweight='bold')
ax.legend(handles=[
    mpatches.Patch(color='#1a7d3a', label=r'$R_0<1$: ningún equilibrio endémico'),
    mpatches.Patch(color='#c0392b', label=r'$R_0<1$: dos equilibrios (biestabilidad)'),
    mpatches.Patch(color='#e8a33d', label=r'$R_0>1$: un equilibrio endémico'),
    plt.Line2D([], [], marker='*', color='white', mec='black', ms=15, ls='',
               label='Escenario base'),
    plt.Line2D([], [], marker='o', color='white', mec='black', ms=8, ls='',
               label=r'$\rho$ de referencia')],
    loc='upper left', fontsize=9.5, framealpha=0.95)
plt.tight_layout()
plt.savefig('mapa_numero_equilibrios.png', dpi=300, bbox_inches='tight')
plt.close('all')

n0, n1, n2 = (M == 0).sum(), (M == 1).sum(), (M == 2).sum()
print(f"  R0>1: {100*n2/M.size:.1f}% | R0<1 sin endemicos: {100*n0/M.size:.1f}% "
      f"| biestable: {100*n1/M.size:.1f}%")

# Límites de la banda biestable en el espacio (rho, gamma_t)
idx_g, idx_r = np.where(M == 1)
if idx_r.size > 0:
    print(f"  Banda biestable: rho en [{rhos[idx_r.min()]:.2f}, {rhos[idx_r.max()]:.2f}], "
          f"gamma_t en [{gts[idx_g.min()]:.3f}, {gts[idx_g.max()]:.3f}]")

# --- Figura: diagrama de bifurcación hacia atrás ---
print("Generando diagrama de bifurcacion...")
bps = np.linspace(0.30, 0.80, 3000)
lowR, lowV, hiR, hiV = [], [], [], []
for bp in bps:
    p = dict(params_biestable, beta_p=bp)
    m = ModeloVapeo(p)
    r = m.calcular_R0()
    eqs = m.contar_equilibrios()['equilibrios']
    if len(eqs) == 2:
        lowR.append(r); lowV.append(eqs[0]['prevalencia'])
        hiR.append(r);  hiV.append(eqs[1]['prevalencia'])
    elif len(eqs) == 1:
        hiR.append(r);  hiV.append(eqs[0]['prevalencia'])
lowR, lowV = np.array(lowR), np.array(lowV)
hiR, hiV = np.array(hiR), np.array(hiV)
Rsn, Vsn = lowR.min(), lowV[lowR.argmin()]

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(hiR, hiV, color='#1f4e9c', lw=2.2, label='Rama endémica estable', zorder=3)
ax.plot(lowR, lowV, color='#c0392b', ls='--', lw=2.2,
        label='Rama endémica inestable', zorder=3)
rr = np.linspace(0.60, 1.20, 400)
ax.plot(rr[rr <= 1], np.zeros((rr <= 1).sum()), color='#1f4e9c', lw=2.2, zorder=3)
ax.plot(rr[rr > 1], np.zeros((rr > 1).sum()), color='#c0392b', ls='--', lw=2.2, zorder=3)
ax.axvspan(Rsn, 1.0, color='#c0392b', alpha=0.07, zorder=0)
ax.axvline(1, color='black', ls=':', lw=1.2)
ax.axvline(Rsn, color='gray', ls=':', lw=1.2)
ax.plot([Rsn], [Vsn], marker='o', ms=7, color='#c0392b', mec='black', mew=1.0, zorder=4)
ax.text(1.01, 6.6, r'$R_0=1$', fontsize=11)
ax.text(Rsn + 0.012, 6.6, r'$R_0^{\,\mathrm{sn}}=%.2f$' % Rsn, fontsize=11)
ax.text((Rsn + 1) / 2, 0.42, 'región de biestabilidad', fontsize=10.5,
        ha='center', color='#8c2f22')
ax.set_xlabel(r'$R_0$', fontsize=12)
ax.set_ylabel(r'Prevalencia de equilibrio  $100\,V^*/N$  (%)', fontsize=12)
ax.set_title('Bifurcación hacia atrás: equilibrios endémicos con ' + r'$R_0<1$',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(alpha=0.3)
ax.set_xlim(0.60, 1.20)
ax.set_ylim(-0.35, 7.2)
plt.tight_layout()
plt.savefig('bifurcacion_hacia_atras.png', dpi=300, bbox_inches='tight')
plt.close('all')
print(f"  Punto de retorno silla-nodo: R0 = {Rsn:.4f}, prevalencia = {Vsn:.2f}%")

# ============================================================================
# CALIBRACIÓN INVERSA DE LA TASA DE TRANSMISIÓN AL CRECIMIENTO OBSERVADO
# (Sección "Reconciliación con datos de Costa Rica: escenarios ajustados")
# ============================================================================

print("\n" + "=" * 80)
print("CALIBRACIÓN INVERSA: TRANSMISIÓN COMPATIBLE CON EL CRECIMIENTO 2021-2025")
print("=" * 80)

# --- Estimación por linealización: crecimiento exponencial 4% -> 13% en 4 años ---
prev_2021, prev_2025, anios = 0.04, 0.13, 4.0
r_crecimiento = np.log(prev_2025 / prev_2021) / anios
Gamma_base = params_base['mu'] + params_base['gamma_t'] + params_base['gamma_p']
R0_lineal = 1.0 + r_crecimiento / Gamma_base
print(f"\nTasa de crecimiento exponencial observada: r = {r_crecimiento:.4f} año^-1")
print(f"Estimación lineal (r = Gamma*(R0-1)):       R0 = {R0_lineal:.2f}")

# --- Problema inverso no lineal: escalar beta y beta_p por un factor común k ---
def _prevalencia_final_escala(k):
    """Prevalencia (fracción) alcanzada a los 4 años partiendo del 4%,
    con beta = k*beta_base y beta_p = k*beta_p_base. Devuelve el residuo
    respecto del 13% observado."""
    p = dict(params_base, beta=k * params_base['beta'],
             beta_p=k * params_base['beta_p'])
    m = ModeloVapeo(p)
    y0 = generar_condicion_inicial(p['N'], p['q'], prev_2021)
    tt = np.linspace(0, anios, 4001)
    s = odeint(m.sistema_EDO, y0, tt)
    return s[-1, 2] / p['N'] - prev_2025

k_estrella = brentq(_prevalencia_final_escala, 1.5, 6.0, xtol=1e-10)
beta_cal = k_estrella * params_base['beta']
beta_p_cal = k_estrella * params_base['beta_p']
modelo_cal = ModeloVapeo(dict(params_base, beta=beta_cal, beta_p=beta_p_cal))
R0_cal = modelo_cal.calcular_R0()
print(f"\nProblema inverso (método de Brent sobre el factor de escala k):")
print(f"  k = {k_estrella:.2f}")
print(f"  beta = {beta_cal:.2f} año^-1,  beta_p = {beta_p_cal:.2f} año^-1")
print(f"  R0 = {R0_cal:.2f}")

# Trayectoria calibrada: pico y equilibrio endémico
y0_cal = generar_condicion_inicial(params_base['N'], params_base['q'], prev_2021)
tt_cal = np.linspace(0, 30, 30001)
sol_cal = odeint(modelo_cal.sistema_EDO, y0_cal, tt_cal)
prev_cal = sol_cal[:, 2] / params_base['N'] * 100
i_pico = int(np.argmax(prev_cal))
print(f"  Pico de prevalencia: {prev_cal[i_pico]:.2f}% en t = {tt_cal[i_pico]:.1f} años "
      f"(hacia {2021 + tt_cal[i_pico]:.0f})")
eq_cal = modelo_cal.resolver_equilibrios_endemicos()
for V_star in eq_cal:
    print(f"  Equilibrio endémico: V* = {V_star:.0f} "
          f"({V_star / params_base['N'] * 100:.2f}% de prevalencia)")

# --- Variante: ajuste de beta con beta_p fijo en su valor base ---
def _prevalencia_final_beta(beta_val):
    p = dict(params_base, beta=beta_val)
    m = ModeloVapeo(p)
    y0 = generar_condicion_inicial(p['N'], p['q'], prev_2021)
    tt = np.linspace(0, anios, 4001)
    s = odeint(m.sistema_EDO, y0, tt)
    return s[-1, 2] / p['N'] - prev_2025

beta_solo = brentq(_prevalencia_final_beta, 0.3, 3.0, xtol=1e-10)
print(f"\nVariante con beta_p fijo en {params_base['beta_p']}:")
print(f"  beta requerido = {beta_solo:.2f} año^-1  "
      f"(transmisión efectiva phi*beta = {params_base['phi'] * beta_solo:.2f} año^-1)")

# ============================================================================
# UMBRALES DE APARICIÓN DE EQUILIBRIOS ENDÉMICOS EN rho (rho_b)
# ============================================================================

print("\n" + "=" * 80)
print("UMBRALES rho_b: APARICIÓN DE EQUILIBRIOS ENDÉMICOS AL AUMENTAR rho")
print("=" * 80)

def umbral_rho_endemicos(params, rho_min=0.121, rho_max=20.0, iteraciones=60):
    """
    Encuentra por bisección el menor valor de rho a partir del cual la
    ecuación cúbica admite raíces positivas (equilibrios endémicos),
    manteniendo los restantes parámetros fijos.
    """
    def hay_endemicos(r):
        m = ModeloVapeo(dict(params, rho=r))
        return len(m.resolver_equilibrios_endemicos()) > 0

    lo, hi = rho_min, rho_max
    if hay_endemicos(lo):
        return lo
    if not hay_endemicos(hi):
        return None
    for _ in range(iteraciones):
        mid = 0.5 * (lo + hi)
        if hay_endemicos(mid):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)

rho_b_base = umbral_rho_endemicos(params_base)
rho_b_cons = umbral_rho_endemicos(params_conservador)
print(f"\nEscenario base:        rho_b = {rho_b_base:.2f} año^-1")
print(f"Escenario conservador: rho_b = {rho_b_cons:.2f} año^-1")
print(f"Prevalencia implícita de la conversión (0.121 / rho_b, base): "
      f"{0.121 / rho_b_base * 100:.2f}%")

# Valor de rho en que a_1 cambia de signo (escenario base)
def _a1_de_rho(r):
    return ModeloVapeo(dict(params_base, rho=r)).coeficientes_cubica()[2]

rho_a1_cero = brentq(_a1_de_rho, 0.5, 3.0, xtol=1e-10)
print(f"Cambio de signo de a_1 (base): rho = {rho_a1_cero:.2f} "
      f"(condición necesaria pero no suficiente para equilibrios endémicos)")

# ============================================================================
# MUESTREO DE MONTE CARLO SOBRE LOS RANGOS DE INCERTIDUMBRE PARAMÉTRICA
# ============================================================================

print("\n" + "=" * 80)
print("MONTE CARLO: DISTRIBUCIÓN DE REGÍMENES SOBRE LOS RANGOS DE INCERTIDUMBRE")
print("=" * 80)

# Rangos del Cuadro de parámetros calibrados. El muestreo es uniforme e
# independiente sobre cada rango, con semilla fija para reproducibilidad.
RANGOS_MC = {
    'beta':    (0.15, 0.22),
    'beta_p':  (0.42, 0.58),
    'gamma_t': (0.12, 0.24),
    'gamma_p': (0.09, 0.18),
    'phi':     (0.55, 0.75),
    'phi_p':   (0.65, 0.85),
    'q':       (0.55, 0.68),
}
SEMILLA_MC = 20260901
N_REPLICAS = 20000

def monte_carlo_regimenes(rho_config, n_replicas=N_REPLICAS, semilla=SEMILLA_MC):
    """
    Clasifica n_replicas configuraciones muestreadas uniformemente sobre
    RANGOS_MC según su régimen: supercrítico (R0 > 1), subcrítico sin
    equilibrios endémicos, o biestable (R0 < 1 con dos equilibrios).

    rho_config: valor fijo de rho (float) o tupla (rho_min, rho_max)
    para muestrearlo uniformemente junto con los demás parámetros.
    """
    rng = np.random.default_rng(semilla)
    conteo = {'supercritico': 0, 'sin_endemicos': 0, 'biestable': 0}
    for _ in range(n_replicas):
        p = dict(params_base)
        for nombre, (lo, hi) in RANGOS_MC.items():
            p[nombre] = rng.uniform(lo, hi)
        p['rho'] = (rho_config if isinstance(rho_config, float)
                    else rng.uniform(*rho_config))
        m = ModeloVapeo(p)
        regimen = m.contar_equilibrios()['regimen']
        if regimen == 'supercritico':
            conteo['supercritico'] += 1
        elif regimen == 'biestable':
            conteo['biestable'] += 1
        else:
            conteo['sin_endemicos'] += 1
    return conteo

print(f"\nMuestreo uniforme e independiente, semilla = {SEMILLA_MC}, "
      f"{N_REPLICAS} réplicas por fila.")
print("Nota: rho no interviene en R0, de modo que ambas filas estiman la misma")
print("probabilidad P(R0 > 1); sus valores difieren solo por ruido de muestreo.\n")

for etiqueta, rho_cfg in [("rho = 0.121 (valor calibrado)", 0.121),
                          ("rho ~ U[0.10, 3.10]", (0.10, 3.10))]:
    c = monte_carlo_regimenes(rho_cfg)
    n = sum(c.values())
    print(f"{etiqueta}:")
    print(f"  R0 > 1:                 {100 * c['supercritico'] / n:.2f}%")
    print(f"  R0 < 1, sin endémicos:  {100 * c['sin_endemicos'] / n:.2f}%")
    print(f"  R0 < 1, biestable:      {100 * c['biestable'] / n:.2f}%")

# ============================================================================
# ROBUSTEZ FRENTE A LA INTERPRETACIÓN DEMOGRÁFICA DE mu
# ============================================================================

print("\n" + "=" * 80)
print("ROBUSTEZ FRENTE A LA INTERPRETACIÓN DE mu (RENOVACIÓN DE COHORTE)")
print("=" * 80)

MUS_ROBUSTEZ = [(0.0125, "1/80 (esperanza de vida)"),
                (0.10,   "1/10 (renovación de cohorte)"),
                (0.20,   "1/5 (renovación de cohorte)")]

def calibrar_factor_transmision(mu_valor, prev_ini=0.04, prev_fin=0.13, anios=4.0):
    """
    Determina el factor de escala k tal que, con beta = k*beta_base y
    beta_p = k*beta_p_base, la trayectoria del sistema pasa de prev_ini a
    prev_fin en el número de años indicado, para el valor de mu dado.

    Permite repetir la calibración inversa bajo distintas interpretaciones
    demográficas del parámetro mu.
    """
    def residuo(k):
        p = dict(params_base, mu=mu_valor,
                 beta=k * params_base['beta'], beta_p=k * params_base['beta_p'])
        m = ModeloVapeo(p)
        y0 = generar_condicion_inicial(p['N'], p['q'], prev_ini)
        s = odeint(m.sistema_EDO, y0, np.linspace(0, anios, 4001))
        return s[-1, 2] / p['N'] - prev_fin
    return brentq(residuo, 1.01, 40.0, xtol=1e-10)

for mu_valor, etiqueta in MUS_ROBUSTEZ:
    print(f"\nmu = {mu_valor}  [{etiqueta}]")
    m_base_mu = ModeloVapeo(dict(params_base, mu=mu_valor))
    print(f"  R0 escenario base           = {m_base_mu.calcular_R0():.4f}")

    k = calibrar_factor_transmision(mu_valor)
    p_cal = dict(params_base, mu=mu_valor,
                 beta=k * params_base['beta'], beta_p=k * params_base['beta_p'])
    m_cal_mu = ModeloVapeo(p_cal)
    print(f"  k calibrado                 = {k:.2f}")
    print(f"  R0 escenario calibrado      = {m_cal_mu.calcular_R0():.4f}")

    eqs_mu = m_cal_mu.resolver_equilibrios_endemicos()
    if eqs_mu:
        V_mu = eqs_mu[0]
        e_mu = m_cal_mu.calcular_equilibrio_completo(V_mu)
        st_mu = m_cal_mu.analizar_estabilidad(
            (e_mu['S'], e_mu['P'], V_mu, e_mu['Qt'], e_mu['Qp']))
        print(f"  Equilibrio endémico         = {100*V_mu/p_cal['N']:.2f}% "
              f"({'estable' if st_mu['max_parte_real'] < 0 else 'inestable'})")
        print(f"  Qp*/V* = gamma_p/mu         = {params_base['gamma_p']/mu_valor:.2f}")
        print(f"  Fracción en abandono        = "
              f"{100*(e_mu['Qt']+e_mu['Qp'])/p_cal['N']:.1f}%")

    y0_mu = generar_condicion_inicial(p_cal['N'], p_cal['q'], 0.04)
    tt_mu = np.linspace(0, 40, 40001)
    pv_mu = odeint(m_cal_mu.sistema_EDO, y0_mu, tt_mu)[:, 2] / p_cal['N'] * 100
    i_mu = int(np.argmax(pv_mu))
    print(f"  Pico calibrado              = {pv_mu[i_mu]:.2f}% en {2021 + tt_mu[i_mu]:.0f}")

    m_mod_mu = ModeloVapeo(dict(params_ajustado_moderado, mu=mu_valor))
    eqm_mu = m_mod_mu.resolver_equilibrios_endemicos()
    print(f"  Escenario moderado          : R0 = {m_mod_mu.calcular_R0():.4f}, "
          f"equilibrio = "
          f"{('%.2f%%' % (100*eqm_mu[0]/10000)) if eqm_mu else 'ninguno'}")

print("\n4. Reconciliación con datos de Costa Rica:")
print(f"   - Escenario base (literatura): R₀ = {resultados_base['R0']:.4f} (subcrítico)")
print(f"   - Escenario ajustado moderado: R₀ = {resultado_moderado['R0']:.4f} (supercrítico)")
print(f"   - Escenario ajustado alto: R₀ = {resultado_alto['R0']:.4f} (supercrítico)")
print("   - Los escenarios ajustados son consistentes con crecimiento 225% observado")
print("   - Sugieren condiciones epidemiológicas específicas de Costa Rica")

print("\n5. Recomendaciones para políticas públicas:")
print("   a) Intervenciones de alta prioridad:")
print("      - Programas de cesación (impacto directo en γt, γp)")
print("      - Regulación de marketing (reducción de β, βp)")
print("      - Campañas con testimonios (mejora de φ, φp)")
print("   b) Intervenciones combinadas son más efectivas que individuales")
print("   c) Monitoreo continuo de prevalencia para ajuste de políticas")
print("   d) Enfoque diferenciado para poblaciones susceptibles vs predispuestas")

print("\n" + "="*80)
print("ANÁLISIS DE SENSIBILIDAD COMPLETADO")
print("="*80)

print("\nARCHIVOS GENERADOS:")
print("  Secciones 5.7-5.8:")
print("    - simulaciones_escenario_base.png")
print("    - simulaciones_escenario_conservador.png")
print("    - comparacion_escenarios_13pct.png")
print("  Sección 5.9:")
print("    - sensibilidad_univariada_completa.png")
print("    - mapa_calor_beta_vs_gamma_t.png")
print("    - mapa_calor_beta_p_vs_gamma_p.png")
print("    - mapa_calor_phi_vs_phi_p.png")
print("    - intervencion_cesacion_mejorada.png")
print("    - intervencion_concientizacion.png")
print("    - intervencion_testimonios_exvapeadores.png")
print("    - intervencion_intervencion_combinada.png")
print("    - comparacion_intervenciones_R0.png")
print("    - escenario_ajustado_moderado.png")
print("    - escenario_ajustado_alto.png")
print("    - comparacion_todos_escenarios_temporal.png")
print("  Sección 5.9.10:")
print("    - mapa_numero_equilibrios.png")
print("    - bifurcacion_hacia_atras.png")

print("\n" + "="*80)
print("FIN DEL ANÁLISIS")
print("="*80)
