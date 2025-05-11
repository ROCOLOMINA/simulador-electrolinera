import streamlit as st
import pandas as pd
import numpy as np
import simpy
import random
from datetime import datetime, timedelta

# ------------------------------------------------------------------------------------
# Diccionarios de traducción para días y meses en español
# ------------------------------------------------------------------------------------

DIAS_SEMANA = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
MESES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
         "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]


# ------------------------------------------------------------------------------------
# 1. Funciones para "parsear" los DataFrames que sube el usuario
# ------------------------------------------------------------------------------------

def parse_configuration_df(config_df):
    """
    A partir de un DataFrame con columnas:
      tipo_cargador, potencia_kW, cantidad, queue_limit (opcional)
    Devuelve:
      CHARGERS (dict)
    """
    chargers = {}

    for _, row in config_df.iterrows():
        tipo = row["tipo_cargador"]
        potencia = float(row["potencia_kW"])
        cantidad = int(row["cantidad"])

        # extraemos queue_limit solo si existe y no es NaN
        queue_limit = None
        if "queue_limit" in config_df.columns and not pd.isna(row["queue_limit"]):
            queue_limit = int(row["queue_limit"])

        # añadimos directamente
        chargers[tipo] = {
            "count": cantidad,
            "power": potencia,
            "queue_limit": queue_limit,
            "precio_venta": float(row["precio_venta"]) if "precio_venta" in config_df.columns else 0.5
        }

    return chargers

def parse_car_types_df(car_types_df):
    """
    A partir de un DataFrame con columnas:
      tipo_coche, probabilidad, bateria_media, bateria_desviacion
    Devuelve un dict CAR_TYPES.
    """
    car_types = {}
    for _, row in car_types_df.iterrows():
        car_types[row["tipo_coche"]] = {
            "probabilidad": row["probabilidad"],
            "bateria_media": row["bateria_media"],
            "bateria_desviacion": row["bateria_desviacion"],
            "count": 0
        }
    return car_types

def parse_simulation_params_df(params_df):
    """
    A partir de un DataFrame con columnas:
      tiempo_simulacion_min, numero_replicaciones
    Devuelve SIM_TIME (int) y NUM_REPLICATIONS (int).
    """
    sim_time = int(params_df.loc[0, "tiempo_simulacion_min"])
    num_reps = int(params_df.loc[0, "numero_replicaciones"])
    start_date = datetime.strptime(params_df.loc[0, "fecha_inicio_simulacion"], "%Y-%m-%d")
    return sim_time, num_reps, start_date

def parse_inter_arrival_df(inter_arrival_df):
    """
    DataFrame con índices (por ejemplo: 0->Lunes, 6->Domingo) y columnas (meses).
    Devuelve el DataFrame y la lista de MONTHS.
    """
    MONTHS = list(inter_arrival_df.columns)
    return inter_arrival_df, MONTHS

# ------------------------------------------------------------------------------------
# 2. Clase Electrolinera y procesos de SimPy
# ------------------------------------------------------------------------------------

class Electrolinera:
    def __init__(self, env, CHARGERS, CAR_TYPES):
        self.env = env
        self.CHARGERS = CHARGERS
        self.CAR_TYPES = CAR_TYPES

        # Estadísticas nuevas
        self.total_wait_time = 0.0                   # sumatorio de todo el tiempo de espera (antes de cargar)
        self.waiting_cars = 0                        # cuántos coches han tenido que esperar
        self.charging_times = []                     # lista de tiempos de carga individuales
        self.energy_by_type = {                      # energía kWh entregada, por tipo de cargador
            ctype: 0.0
            for ctype in CHARGERS.keys()
        }

        self.stations = []
        # Crear los cargadores
        for charger_type, config in CHARGERS.items():
            for _ in range(config["count"]):
                self.stations.append({
                    "type": charger_type,
                    "resource": simpy.Resource(env, capacity=1),
                    "power": config["power"],
                    "queue_limit": config["queue_limit"],
                    "busy_time": 0.0
                })

        # Variables para estadísticas
        self.charger_queue_times = []
        self.queue_lengths = []

    def charge(self, car, station, car_type, frac_carga):
        car_capacity = float(car_type[:-3])     # kWh
        charger_power = station["power"]        # kW

        # distribuye tiempo (minutos)
        mean = frac_carga * (car_capacity / charger_power) * 60
        std = self.CAR_TYPES[car_type]["bateria_desviacion"] * (car_capacity / charger_power) * 60
        charging_time = max(1, np.random.normal(loc=mean, scale=std))

        # energía entregada en kWh
        energy_delivered = frac_carga * car_capacity

        yield self.env.timeout(charging_time)
        return charging_time, energy_delivered

# ------------------------------------------------------------------------------------
# 3. Funciones de llegada de coches y simulación con SimPy
# ------------------------------------------------------------------------------------

def get_inter_arrival_time(current_time, inter_arrival_df, MONTHS):
    """
    Devuelve el tiempo medio de llegada en minutos, según la fecha (mes y día de la semana).
    """
    month = MESES[current_time.month - 1]                # Mes en español
    weekday_name = DIAS_SEMANA[current_time.weekday()]   # Día en español
    return inter_arrival_df.loc[weekday_name, month]

def car(env, name, electrolinera, stats, precio_dia_mes_df,
        energy_by_car_type, energy_by_weekday, energy_by_month, start_datetime,
        served_by_car_type, abandoned_by_car_type):
    """
    Proceso de cada coche:
      - Selecciona el tipo de coche según probabilidades.
      - Calcula la fracción de batería a cargar.
      - Hace cola en un cargador y se va al terminar la carga.
    """
    arrival_time = env.now
    queue_length = sum(len(station["resource"].queue) for station in electrolinera.stations)
    electrolinera.queue_lengths.append(queue_length)

    car_type = random.choices(
        list(electrolinera.CAR_TYPES.keys()),
        weights=[electrolinera.CAR_TYPES[key]["probabilidad"] for key in electrolinera.CAR_TYPES]
    )[0]
    electrolinera.CAR_TYPES[car_type]["count"] += 1
    # Guardar el tipo en la variable stats
    stats["car_type"] = car_type

    media_bateria = electrolinera.CAR_TYPES[car_type]["bateria_media"]
    desv_bateria = electrolinera.CAR_TYPES[car_type]["bateria_desviacion"]
    frac_carga = np.random.normal(media_bateria, desv_bateria)
    frac_carga = np.clip(frac_carga, 0.01, 1.0)

    available_stations = [
        s for s in electrolinera.stations
        if s["queue_limit"] is None or len(s["resource"].queue) < s["queue_limit"]
    ]
    if not available_stations:
        stats["abandoned"] += 1
        abandoned_by_car_type[car_type] += 1
        stats["system_times"].append(0)
        return


    station = min(available_stations, key=lambda s: len(s["resource"].queue))

    with station["resource"].request() as req:
        # 1) mido espera en cola
        wait_start = env.now
        yield req
        wait_time = env.now - wait_start

        # acumular totales
        electrolinera.charger_queue_times.append(wait_time)
        electrolinera.total_wait_time += wait_time
        if wait_time > 0:
            electrolinera.waiting_cars += 1

        # 2) arranca la carga
        charge_start = env.now
        charging_time, energy = yield env.process(
            electrolinera.charge(name, station, car_type, frac_carga)
        )

        energy_by_car_type[car_type] += energy



        fecha = start_datetime + timedelta(minutes=env.now)
        mes_nombre = MESES[fecha.month - 1]              # "Enero", "Febrero", …
        dia_nombre = DIAS_SEMANA[fecha.weekday()]        # "Lunes", "Martes", …

        precio = precio_dia_mes_df.loc[dia_nombre, mes_nombre]

        stats["coste_energia"] += energy * precio
        energy_by_weekday[fecha.weekday()] += energy

        # si tus claves van de 1 a 12:
        energy_by_month[fecha.month] += energy

        charge_end = env.now

        # estadísticas de ocupación y energía
        station["busy_time"] += (charge_end - charge_start)
        electrolinera.charging_times.append(charging_time)
        electrolinera.energy_by_type[station["type"]] += energy

    # Al terminar la carga, damos el coche por servido y registramos su tiempo total
    stats["served"] += 1
    served_by_car_type[car_type] += 1
    stats["system_times"].append(charge_end - arrival_time)

def car_generator(
    env, electrolinera, stats,
    inter_arrival_df, MONTHS,
    SIM_TIME, precio_dia_mes_df,
    energy_by_car_type, energy_by_weekday, energy_by_month, start_datetime,
    served_by_car_type, abandoned_by_car_type
):

    """
    Genera coches de forma continua durante el tiempo de simulación.
    """
    i = 0
    current_time = datetime(2024, 1, 1)
    while True:
        iti = get_inter_arrival_time(current_time, inter_arrival_df, MONTHS)
        yield env.timeout(random.expovariate(1 / iti))

        # lanzamos un proceso 'car'
        env.process(
            car(
                env,
                f"Car {i}",
                electrolinera,
                stats,
                precio_dia_mes_df,
                energy_by_car_type,
                energy_by_weekday,
                energy_by_month,
                start_datetime,
                served_by_car_type,
                abandoned_by_car_type
            )
        )

        current_time += timedelta(minutes=iti)
        i += 1

def run_simulation_from_dfs(
    config_df, car_types_df, params_df,
    inter_arrival_df, econ_params_df, precio_dia_mes_df
):

    """
    Función principal que toma los DataFrames (o CSV subidos por el usuario),
    ejecuta la simulación y devuelve un DataFrame con los resultados.
    """
    CHARGERS = parse_configuration_df(config_df)
    CAR_TYPES = parse_car_types_df(car_types_df)
    SIM_TIME, NUM_REPLICATIONS, START_DATE = parse_simulation_params_df(params_df)
    inter_arrival_parsed, MONTHS = parse_inter_arrival_df(inter_arrival_df)

    def run_simulation(replication):
        # Fijar semillas para reproducibilidad
        random.seed(42 + replication)
        np.random.seed(42 + replication)

        # — SUPUESTOS ECONÓMICOS —
        economics = dict(zip(econ_params_df["variable"], econ_params_df["valor"]))

        coste_mantenimiento_pct = float(economics["coste_mantenimiento_pct"])
        vida_util_cargador = float(economics["vida_util_cargador"])
        coste_explotacion_anual = float(economics["coste_explotacion_anual"])
        coste_fijo = float(economics["coste_fijo"])
        coste_variable = float(economics["coste_variable"])

        # Fórmula de coste de cargador en función de su potencia
        def calcular_coste_cargador(potencia_kW):
            return coste_fijo + coste_variable * potencia_kW

        # Crear entorno y electrolinera
        env = simpy.Environment()
        electrolinera = Electrolinera(
            env,
            CHARGERS,
            CAR_TYPES
        )

        # Acumuladores detallados
        energy_by_car_type = {tipo: 0.0 for tipo in CAR_TYPES}
        served_by_car_type = {tipo: 0 for tipo in CAR_TYPES}
        abandoned_by_car_type = {tipo: 0 for tipo in CAR_TYPES}

        energy_by_weekday = {i: 0.0 for i in range(7)}       # 0=Lunes, 6=Domingo
        energy_by_month = {m: 0.0 for m in range(1, 13)}     # 1=Enero, ..., 12=Diciembre

        

        # Estadísticas básicas
        stats = {"served": 0, "abandoned": 0, "system_times": [], "coste_energia": 0.0}

# Arrancamos el generador de llegadas
        env.process(
        car_generator(
            env,
            electrolinera,
            stats,
            inter_arrival_parsed,
            MONTHS,
            SIM_TIME,
            precio_dia_mes_df,
            energy_by_car_type,
            energy_by_weekday,
            energy_by_month,
            START_DATE,
            served_by_car_type,
            abandoned_by_car_type
        )
    )



        # Ejecutar simulación
        env.run(until=SIM_TIME)

        # Conteo final por tipo de coche
        car_type_counts = {
            key: electrolinera.CAR_TYPES[key]["count"]
            for key in electrolinera.CAR_TYPES
        }

        # — CÁLCULO DE NUEVAS MÉTRICAS —
        sim_hours = SIM_TIME / 60.0
        num_chargers = sum(cfg["count"] for cfg in CHARGERS.values())
        days = SIM_TIME / (24 * 60)

        # 1) Espera
        mean_wait = (
            electrolinera.total_wait_time / stats["served"]
            if stats["served"] else 0
        )
        pct_wait = (
            electrolinera.waiting_cars / stats["served"] * 100
            if stats["served"] else 0
        )

        # 2) Carga
        mean_charge_time = (
            np.mean(electrolinera.charging_times)
            if electrolinera.charging_times else 0
        )

        # 3) Ocupación
        total_busy = sum(st["busy_time"] for st in electrolinera.stations)
        utilization = total_busy / (num_chargers * SIM_TIME) * 100

        # 4) Energía por cargador y hora
        total_energy = sum(electrolinera.energy_by_type.values())
        energy_per_charger_per_hour = total_energy / num_chargers / sim_hours

        # — ECONOMÍA SENCILLA —

        # 1) Ingresos por venta de energía
        ingresos = sum(
            electrolinera.energy_by_type[cargador] * CHARGERS[cargador]["precio_venta"]
            for cargador in CHARGERS
        )

        # 2) Coste de electricidad comprada
        coste_energia = stats["coste_energia"]
        start_datetime = datetime(2024, 1, 1)  # inicio de la simulación

        # 3) Coste de mantenimiento de todos los cargadores
        coste_mantenimiento = sum(
            cfg["count"] * calcular_coste_cargador(cfg["power"]) * coste_mantenimiento_pct
            for cfg in CHARGERS.values()
        )

        # 4) Coste de amortización de cargadores
        coste_amortizacion = sum(
            cfg["count"] * calcular_coste_cargador(cfg["power"]) / vida_util_cargador
            for cfg in CHARGERS.values()
        )

        # 5) Costes totales
        costes_totales = coste_energia + coste_mantenimiento + coste_amortizacion + coste_explotacion_anual

        # 6) Beneficio neto
        beneficio_neto = ingresos - costes_totales

        # 7) Inversión inicial (CAPEX)
        inversion_inicial = sum(
            cfg["count"] * calcular_coste_cargador(cfg["power"])
            for cfg in CHARGERS.values()
        )

        # 8) Rentabilidad
        rentabilidad = (beneficio_neto / inversion_inicial) * 100 if inversion_inicial > 0 else 0

        # 5) Coches atendidos por día
        cars_per_day = stats["served"] / days

        dias_semana = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
        meses = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
                 "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]

        served_counts = {f"Coches atendidos tipo {tipo}": count for tipo, count in served_by_car_type.items()}
        abandoned_counts = {f"Coches que abandonan tipo {tipo}": count for tipo, count in abandoned_by_car_type.items()}


        # — DEVOLUCIÓN DE TODAS LAS MÉTRICAS —
        return {
            # Métricas originales
            "Coches atendidos": stats["served"],
            "Coches que abandonan sin servicio": stats["abandoned"],
            "Tasa de abandono (%)": (
                stats["abandoned"] / (stats["served"] + stats["abandoned"]) * 100
                if (stats["served"] + stats["abandoned"]) > 0 else 0
            ),
            "Tiempo promedio en el sistema (min)": (
                np.mean(stats["system_times"]) if stats["system_times"] else 0
            ),
            "Tiempo máximo en el sistema (min)": max(stats["system_times"], default=0),
            "Tiempo mínimo en el sistema (min)": min(stats["system_times"], default=0),
            "Tiempo promedio en cola de cargadores (min)": (
                np.mean(electrolinera.charger_queue_times)
                if electrolinera.charger_queue_times else 0
            ),
            "Tiempo máximo en cola de cargadores (min)": max(electrolinera.charger_queue_times, default=0),
            "Tiempo mínimo en cola de cargadores (min)": min(electrolinera.charger_queue_times, default=0),
            "Longitud promedio de la cola": (
                np.mean(electrolinera.queue_lengths) if electrolinera.queue_lengths else 0
            ),
            "Longitud máxima de la cola": max(electrolinera.queue_lengths, default=0),
            # Conteo por tipo de coche
            **car_type_counts,
            **served_counts,
            **abandoned_counts,

            # Métricas nuevas
            "Tiempo medio de espera (min)": mean_wait,
            "% Coches que esperan": pct_wait,
            "Tiempo medio de carga (min)": mean_charge_time,
            "Ocupación media cargadores (%)": utilization,
            "Energía/cargador/hora (kWh)": energy_per_charger_per_hour,
            "Coches atendidos/día": cars_per_day,

            # Energía total por tipo de cargador
            **{
                f"Energía total {ctype} (kWh)": energy
                for ctype, energy in electrolinera.energy_by_type.items()
            },

            # Métricas economia
            "Ingresos anuales (€)": ingresos,
            "Costes anuales (€)": costes_totales,
            "Beneficio neto anual (€)": beneficio_neto,
            "Rentabilidad sobre inversión (%)": rentabilidad,

            # Energía total por tipo de coche
            **{f"Energía total coche {tipo} (kWh)": energia for tipo, energia in energy_by_car_type.items()},
            # Energía total por día de la semana (con nombres)
            **{f"Energía total {dias_semana[i]} (kWh)": energia for i, energia in energy_by_weekday.items()},

            # Energía total por mes (con nombres)
            **{f"Energía total {meses[m - 1]} (kWh)": energia for m, energia in energy_by_month.items()},

        }

    results = []
    for rep in range(NUM_REPLICATIONS):
        results.append(run_simulation(rep))
    df_results = pd.DataFrame(results)
    return df_results

# ------------------------------------------------------------------------------------
# 4. Interfaz de usuario con Streamlit
# ------------------------------------------------------------------------------------

def main():
    st.title("Simulador de Electrolinera")
    st.write("Bienvenido/a a la aplicación de simulación. Sube tus CSV o utiliza los datos de ejemplo.")

    st.subheader("1) Configuración de Cargadores (precio en euros)")
    config_file = st.file_uploader("Sube 'configuracion_cargadores.csv' ", type=["csv"])
    if config_file is not None:
        config_df = pd.read_csv(config_file)
        st.dataframe(config_df)
    else:
        st.info("Usando configuración de ejemplo.")
        config_df = pd.DataFrame({
            "tipo_cargador": ["100kW", "350kW"],
            "potencia_kW": [100, 350],
            "cantidad": [1, 1],
            "queue_limit": [1, 1],
            "precio_venta": [0.45,0.55]
        })
        st.dataframe(config_df)

    st.subheader("2) Tipos de Coches")
    cars_file = st.file_uploader("Sube 'tipos_coches.csv'", type=["csv"])
    if cars_file is not None:
        car_types_df = pd.read_csv(cars_file)
        st.dataframe(car_types_df)
    else:
        st.info("Usando tipos de coches de ejemplo.")
        car_types_df = pd.DataFrame({
            "tipo_coche": ["120kWh", "70kWh", "40kWh"],
            "probabilidad": [0.2, 0.3, 0.5],
            "bateria_media": [0.6, 0.6, 0.6],
            "bateria_desviacion": [0.2, 0.2, 0.2]
        })
        st.dataframe(car_types_df)

    st.subheader("3) Parámetros de Simulación")
    params_file = st.file_uploader("Sube 'parametros_simulacion.csv'", type=["csv"])
    if params_file is not None:
        params_df = pd.read_csv(params_file)
        st.dataframe(params_df)
    else:
        st.info("Usando parámetros de ejemplo.")
        params_df = pd.DataFrame({
            "tiempo_simulacion_min": [525600],
            "numero_replicaciones": [1],
            "fecha_inicio_simulacion": ["2030-01-01"]
        })
        st.dataframe(params_df)

    st.subheader("4) Tiempos de Llegada (tiempo_entre_llegadas.csv) (minutos entre llegada de coches)")
    inter_file = st.file_uploader("Sube 'tiempo_entre_llegadas.csv'", type=["csv"])

    if inter_file is not None:
        inter_arrival_df = pd.read_csv(inter_file, index_col=0)
        st.dataframe(inter_arrival_df)
    else:
        st.info("Usando tiempos de llegada de ejemplo.")
        inter_arrival_df = pd.DataFrame({
            'Enero':     [10.015, 10.009, 9.925, 10.442, 7.793, 12.118, 14.094],
            'Febrero':   [8.648, 9.637, 9.383, 8.274, 6.288, 10.572, 12.811],
            'Marzo':     [10.657, 10.127, 10.110, 9.292, 6.941, 11.133, 13.279],
            'Abril':     [8.498, 8.365, 7.140, 6.828, 6.645, 10.373, 10.516],
            'Mayo':      [9.786, 9.091, 9.034, 8.031, 5.786, 9.334, 11.812],
            'Junio':     [8.650, 8.987, 8.570, 7.637, 5.632, 8.884, 11.082],
            'Julio':     [8.067, 8.154, 8.086, 6.829, 5.302, 6.786, 9.329],
            'Agosto':    [7.038, 7.992, 8.115, 7.495, 5.710, 7.907, 10.169],
            'Septiembre':[9.417, 9.227, 9.118, 8.120, 6.035, 9.613, 11.541],
            'Octubre':   [9.249, 9.180, 9.063, 8.385, 5.974, 8.763, 8.712],
            'Noviembre': [9.562, 9.519, 9.508, 8.961, 6.735, 10.611, 11.845],
            'Diciembre': [9.244, 9.089, 8.677, 8.355, 6.241, 9.412, 10.608]
        }, index=["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"])

        st.dataframe(inter_arrival_df)

    st.subheader("5) Supuestos Económicos (euros o años)")
    econ_file = st.file_uploader("Sube 'parametros_economicos.csv'", type=["csv"])
    if econ_file is not None:
        econ_params_df = pd.read_csv(econ_file)
        st.dataframe(econ_params_df)
    else:
        st.info("Usando supuestos económicos de ejemplo.")
        econ_params_df = pd.DataFrame({
            "variable": [
                "coste_mantenimiento_pct",
                "vida_util_cargador", "coste_explotacion_anual",
                "coste_fijo", "coste_variable"
            ],
            "valor": [
                0.07, 15, 50000,
                20000, 300
            ]
        })
    st.dataframe(econ_params_df)

    st.subheader("6) Precio de compra por día y mes (€/kWh)")
    precio_dia_mes_file = st.file_uploader("Sube 'precio_compra_por_dia_y_mes.csv'", type=["csv"])
    if precio_dia_mes_file is not None:
        precio_dia_mes_df = pd.read_csv(precio_dia_mes_file, index_col=0)
        st.dataframe(precio_dia_mes_df)
    else:
        st.info("Usando precios de compra por día y mes de ejemplo.")
        precio_dia_mes_df = pd.DataFrame({
            "Enero":     [0.145, 0.144, 0.148, 0.152, 0.222, 0.189, 0.123],
            "Febrero":   [0.158, 0.161, 0.175, 0.198, 0.219, 0.185, 0.134],
            "Marzo":     [0.179, 0.179, 0.184, 0.202, 0.221, 0.162, 0.141],
            "Abril":     [0.201, 0.198, 0.195, 0.217, 0.214, 0.174, 0.162],
            "Mayo":      [0.223, 0.225, 0.224, 0.243, 0.236, 0.193, 0.179],
            "Junio":     [0.181, 0.176, 0.185, 0.200, 0.199, 0.167, 0.158],
            "Julio":     [0.164, 0.165, 0.167, 0.172, 0.176, 0.153, 0.144],
            "Agosto":    [0.142, 0.143, 0.145, 0.151, 0.157, 0.135, 0.124],
            "Septiembre":[0.132, 0.133, 0.134, 0.137, 0.140, 0.124, 0.113],
            "Octubre":   [0.153, 0.155, 0.156, 0.159, 0.161, 0.135, 0.122],
            "Noviembre": [0.172, 0.174, 0.176, 0.179, 0.181, 0.153, 0.131],
            "Diciembre": [0.193, 0.195, 0.197, 0.199, 0.202, 0.174, 0.147]
    }, index=["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"])

    st.dataframe(precio_dia_mes_df)

    if st.button("Ejecutar Simulación"):
        st.write("Ejecutando simulación, por favor espera...")
        results_df = run_simulation_from_dfs(config_df, car_types_df, params_df, inter_arrival_df, econ_params_df, precio_dia_mes_df)
        st.success("Simulación finalizada.")

        # 1) DataFrame completo
        st.subheader("Resultados por réplica")
        st.dataframe(results_df)

        # 2) Resumen de métricas (promedio sobre réplicas)
        st.subheader("Resumen global (media sobre réplicas)")
        summary_df = results_df.mean().to_frame(name="Promedio").round(2)
        st.table(summary_df)

        # 3) Botón de descarga de CSV con todos los datos
        csv_result = results_df.to_csv(index=False)
        st.download_button(
            label="Descargar resultados completos",
            data=csv_result,
            file_name="resultados_simulacion.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()


