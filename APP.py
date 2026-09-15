# -*- coding: utf-8 -*-
"""
Taller 1 - Inteligencia Artificial
Aplicativo web con los 4 algoritmos genéticos (interfaz Streamlit).
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Taller Algoritmos Genéticos", layout="centered")
st.title("Taller 1 - Algoritmos Genéticos")
st.caption("Selecciona una pestaña para probar cada problema.")

tab1, tab2, tab3, tab4 = st.tabs(["N-Reinas", "TSP", "Asignación de cursos", "Mochila"])

# ============================================================
# TAB 1: N-REINAS
# ============================================================
with tab1:
    st.header("Problema base: N-Reinas")

    n = st.slider("Número de reinas (N)", 4, 12, 8, key="n_reinas")
    pob = st.slider("Tamaño de población", 20, 200, 100, key="pob_reinas")
    mut = st.select_slider("Tasa de mutación", options=[0.05, 0.1, 0.2], value=0.1, key="mut_reinas")

    def fitness_reinas(board):
        n = len(board)
        ataques = sum(1 for i in range(n) for j in range(i + 1, n)
                      if board[i] == board[j] or abs(board[i] - board[j]) == abs(i - j))
        return -ataques

    def ga_reinas(n, pob, mut, generaciones=300):
        poblacion = [np.random.permutation(n) for _ in range(pob)]
        historial = []
        for gen in range(generaciones):
            fit = np.array([fitness_reinas(ind) for ind in poblacion])
            historial.append(max(fit))
            if max(fit) == 0:
                return poblacion[np.argmax(fit)], historial, gen + 1
            padres = [poblacion[i] for i in np.argsort(fit)[-pob // 2:]]
            nueva = []
            for _ in range(pob):
                p1, p2 = padres[np.random.randint(len(padres))], padres[np.random.randint(len(padres))]
                punto = np.random.randint(1, n - 1)
                hijo = np.concatenate((p1[:punto], p2[punto:]))
                faltantes = list(set(range(n)) - set(hijo))
                np.random.shuffle(faltantes)
                for i in range(n):
                    if list(hijo).count(hijo[i]) > 1:
                        hijo[i] = faltantes.pop()
                if np.random.rand() < mut:
                    i, j = np.random.randint(0, n, size=2)
                    hijo[i], hijo[j] = hijo[j], hijo[i]
                nueva.append(hijo)
            poblacion = nueva
        return None, historial, None

    if st.button("Ejecutar N-Reinas"):
        sol, hist, gen = ga_reinas(n, pob, mut)
        if sol is not None:
            st.success(f"Solución encontrada en la generación {gen}")
            st.write("Tablero solución:", list(sol))
        else:
            st.warning("No se encontró solución sin ataques en el límite de generaciones.")

        fig, ax = plt.subplots()
        ax.plot(hist, marker='o', color='b')
        ax.set_xlabel("Generaciones")
        ax.set_ylabel("Fitness (0 = sin ataques)")
        ax.set_title(f"Convergencia N-Reinas (N={n})")
        ax.grid(True)
        st.pyplot(fig)

# ============================================================
# TAB 2: TSP
# ============================================================
with tab2:
    st.header("Ejercicio 1: Agente Viajero (TSP)")

    num_ciudades = st.slider("Número de ciudades", 5, 20, 10, key="n_ciudades")
    mut_tsp = st.select_slider("Tasa de mutación", options=[0.05, 0.1, 0.2], value=0.1, key="mut_tsp")

    def distancia_ruta(ruta, dist):
        return sum(dist[ruta[i], ruta[(i + 1) % len(ruta)]] for i in range(len(ruta)))

    def cruce_ox(p1, p2):
        n = len(p1)
        i, j = sorted(np.random.choice(range(n), size=2, replace=False))
        hijo = [-1] * n
        hijo[i:j] = p1[i:j]
        pos = j
        for gen in list(p2[j:]) + list(p2[:j]):
            if gen not in hijo:
                if pos >= n:
                    pos = 0
                hijo[pos] = gen
                pos += 1
        return np.array(hijo)

    def ga_tsp(dist, pob=100, generaciones=300, mut=0.1, elite=2):
        n = dist.shape[0]
        poblacion = [np.random.permutation(n) for _ in range(pob)]
        historial = []
        for gen in range(generaciones):
            fit = np.array([1 / (distancia_ruta(ind, dist) + 1e-6) for ind in poblacion])
            historial.append(distancia_ruta(poblacion[np.argmax(fit)], dist))
            elite_idx = np.argsort(fit)[-elite:]
            nueva = [poblacion[i] for i in elite_idx]
            while len(nueva) < pob:
                i1, i2 = np.random.choice(pob, size=2, replace=False)
                p1 = poblacion[i1] if fit[i1] > fit[i2] else poblacion[i2]
                i1, i2 = np.random.choice(pob, size=2, replace=False)
                p2 = poblacion[i1] if fit[i1] > fit[i2] else poblacion[i2]
                hijo = cruce_ox(p1, p2)
                if np.random.rand() < mut:
                    i, j = np.random.randint(0, n, size=2)
                    hijo[i], hijo[j] = hijo[j], hijo[i]
                nueva.append(hijo)
            poblacion = nueva[:pob]
        fit = np.array([1 / (distancia_ruta(ind, dist) + 1e-6) for ind in poblacion])
        mejor = poblacion[np.argmax(fit)]
        return mejor, distancia_ruta(mejor, dist), historial

    if st.button("Ejecutar TSP"):
        np.random.seed(1)
        coords = np.random.uniform(0, 100, (num_ciudades, 2))
        dist = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)

        mejor_ruta, mejor_dist, hist = ga_tsp(dist, mut=mut_tsp)
        st.success(f"Distancia total: {mejor_dist:.2f}")
        st.write("Mejor ruta:", list(mejor_ruta))

        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots()
            ax1.plot(hist, color='b')
            ax1.set_xlabel("Generaciones")
            ax1.set_ylabel("Distancia")
            ax1.set_title("Convergencia")
            ax1.grid(True)
            st.pyplot(fig1)
        with col2:
            fig2, ax2 = plt.subplots()
            ruta_cerrada = list(mejor_ruta) + [mejor_ruta[0]]
            ax2.plot(coords[ruta_cerrada, 0], coords[ruta_cerrada, 1], 'o-', color='green')
            for i, (x, y) in enumerate(coords):
                ax2.text(x, y, str(i))
            ax2.set_title("Ruta óptima")
            ax2.grid(True)
            st.pyplot(fig2)

# ============================================================
# TAB 3: ASIGNACIÓN DE CURSOS
# ============================================================
with tab3:
    st.header("Ejercicio 2: Asignación de cursos a salas")

    CURSOS = [
        {"nombre": "Algoritmos", "est": 25, "pc": True, "sw": True},
        {"nombre": "Bases de Datos", "est": 30, "pc": True, "sw": True},
        {"nombre": "Redes", "est": 20, "pc": True, "sw": False},
        {"nombre": "Cálculo", "est": 40, "pc": False, "sw": False},
        {"nombre": "IA", "est": 22, "pc": True, "sw": True},
        {"nombre": "Física", "est": 35, "pc": False, "sw": False},
        {"nombre": "Sistemas Op.", "est": 18, "pc": True, "sw": False},
        {"nombre": "Diseño Web", "est": 15, "pc": True, "sw": True},
    ]
    SALAS = [
        {"nombre": "Sala 1", "cap": 20, "pc": True, "sw": True},
        {"nombre": "Sala 2", "cap": 35, "pc": True, "sw": False},
        {"nombre": "Sala 3", "cap": 45, "pc": False, "sw": False},
        {"nombre": "Sala 4", "cap": 25, "pc": True, "sw": True},
    ]
    FRANJAS = ["7-9am", "9-11am", "11-1pm", "2-4pm", "4-6pm"]
    NF = len(FRANJAS)
    NC = len(CURSOS)
    usar_elite = st.checkbox("Usar elitismo", value=True)

    def penal(ind):
        total, ocup = 0, {}
        for k, gen in enumerate(ind):
            s, f = gen // NF, gen % NF
            c, sa = CURSOS[k], SALAS[s]
            if c["est"] > sa["cap"]:
                total += 10
            if c["pc"] and not sa["pc"]:
                total += 15
            if c["sw"] and not sa["sw"]:
                total += 15
            ocup.setdefault((s, f), []).append(k)
        for v in ocup.values():
            if len(v) > 1:
                total += 20 * (len(v) - 1)
        return total

    def ga_cursos(pob=50, generaciones=200, mut=0.1, elite=usar_elite):
        poblacion = [np.random.randint(0, len(SALAS) * NF, size=NC) for _ in range(pob)]
        for gen in range(generaciones):
            fit = np.array([-penal(ind) for ind in poblacion])
            mejor_idx = np.argmax(fit)
            nueva = [poblacion[mejor_idx]] if elite else []
            while len(nueva) < pob:
                i1, i2 = np.random.choice(pob, size=2, replace=False)
                p1 = poblacion[i1] if fit[i1] > fit[i2] else poblacion[i2]
                i1, i2 = np.random.choice(pob, size=2, replace=False)
                p2 = poblacion[i1] if fit[i1] > fit[i2] else poblacion[i2]
                punto = np.random.randint(1, NC - 1)
                hijo = np.concatenate((p1[:punto], p2[punto:])).copy()
                for i in range(NC):
                    if np.random.rand() < mut:
                        hijo[i] = np.random.randint(0, len(SALAS) * NF)
                nueva.append(hijo)
            poblacion = nueva[:pob]
        fit = np.array([-penal(ind) for ind in poblacion])
        return poblacion[np.argmax(fit)]

    if st.button("Ejecutar Asignación de cursos"):
        mejor = ga_cursos()
        st.success(f"Penalización final: {penal(mejor)}")
        filas = []
        for k, gen in enumerate(mejor):
            s, f = gen // NF, gen % NF
            filas.append([CURSOS[k]["nombre"], SALAS[s]["nombre"], FRANJAS[f]])
        st.table(filas)

# ============================================================
# TAB 4: MOCHILA
# ============================================================
with tab4:
    st.header("Ejercicio 3: Mochila (Knapsack)")

    PESOS = np.array([2, 3, 4, 5, 9, 7, 1, 6, 8, 3, 4, 5, 2, 7, 6])
    VALORES = np.array([3, 4, 5, 8, 10, 6, 1, 9, 7, 4, 5, 6, 2, 8, 7])

    capacidad = st.slider("Capacidad de la mochila", 10, 50, 20, key="cap_mochila")
    metodo = st.radio("Método", ["penalizacion", "reparacion"], horizontal=True)

    def reparar(ind, cap):
        ind = ind.copy()
        sel = np.where(ind == 1)[0]
        np.random.shuffle(sel)
        peso = np.sum(PESOS[ind == 1])
        i = 0
        while peso > cap and i < len(sel):
            ind[sel[i]] = 0
            peso -= PESOS[sel[i]]
            i += 1
        return ind

    def fit_mochila(ind, cap, metodo):
        if metodo == "reparacion":
            ind = reparar(ind, cap)
        peso, valor = np.sum(PESOS[ind == 1]), np.sum(VALORES[ind == 1])
        if metodo == "penalizacion" and peso > cap:
            return valor - 3 * (peso - cap)
        return valor

    def ga_mochila(cap, metodo, pob=30, generaciones=150, mut=0.1):
        n = len(PESOS)
        poblacion = np.random.randint(2, size=(pob, n))
        historial = []
        for gen in range(generaciones):
            fit = [fit_mochila(ind, cap, metodo) for ind in poblacion]
            mejor = poblacion[np.argmax(fit)]
            if metodo == "reparacion":
                mejor = reparar(mejor, cap)
            historial.append(np.sum(VALORES[mejor == 1]))
            i1, i2 = np.random.choice(pob, size=2, replace=False)
            p1 = poblacion[i1] if fit[i1] > fit[i2] else poblacion[i2]
            i1, i2 = np.random.choice(pob, size=2, replace=False)
            p2 = poblacion[i1] if fit[i1] > fit[i2] else poblacion[i2]
            punto = np.random.randint(1, n - 1)
            h1 = np.concatenate((p1[:punto], p2[punto:]))
            h2 = np.concatenate((p2[:punto], p1[punto:]))
            for h in (h1, h2):
                for i in range(n):
                    if np.random.rand() < mut:
                        h[i] = 1 - h[i]
            peores = np.argsort(fit)[:2]
            poblacion[peores[0]], poblacion[peores[1]] = h1, h2
        final = poblacion[np.argmax([fit_mochila(ind, cap, metodo) for ind in poblacion])]
        if metodo == "reparacion":
            final = reparar(final, cap)
        return final, historial

    if st.button("Ejecutar Mochila"):
        mejor, hist = ga_mochila(capacidad, metodo)
        peso, valor = np.sum(PESOS[mejor == 1]), np.sum(VALORES[mejor == 1])
        st.success(f"Valor: {valor} | Peso: {peso} / {capacidad}")
        st.write("Objetos seleccionados:", [i + 1 for i in range(len(mejor)) if mejor[i] == 1])

        fig, ax = plt.subplots()
        ax.plot(hist, color='b')
        ax.set_xlabel("Generaciones")
        ax.set_ylabel("Mejor valor")
        ax.set_title(f"Convergencia ({metodo}, capacidad={capacidad})")
        ax.grid(True)
        st.pyplot(fig)