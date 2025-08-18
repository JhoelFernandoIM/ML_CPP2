import numpy as np
import os
import pandas as pd
import random
from datetime import datetime, timedelta

# ---------------------------
# Parámetros generales

# Crear carpeta si no existe
os.makedirs('data_sintetica', exist_ok=True)

BASE_DIR =os.path.abspath(os.path.join(os.getcwd()))
# ---------------------------
np.random.seed(42)
n_clientes = 20000
hoy = datetime.today()

# ---------------------------
# ID único
# ---------------------------
ids = [f"C-{str(i).zfill(6)}" for i in range(1, n_clientes+1)]

# ---------------------------
# Fechas: primera y última compra
# ---------------------------
# Fecha primera compra: últimos 36 meses, más denso en meses recientes
primeras_compras = [
    hoy - timedelta(days=int(np.random.exponential(scale=400)))
    for _ in range(n_clientes)
]

# Fecha última compra: depende de activo o inactivo
recencia_dias = np.random.exponential(scale=90, size=n_clientes).astype(int)
recencia_dias = np.clip(recencia_dias, 0, 365)
ultimas_compras = [hoy - timedelta(days=int(r)) for r in recencia_dias]

# ---------------------------
# Variables categóricas
# ---------------------------
paises = np.random.choice(["Perú", "Chile", "Colombia", "México", "Otros"], size=n_clientes, p=[0.35,0.2,0.15,0.15,0.15])
canales = np.random.choice(["web", "móvil", "email", "orgánico", "ads"], size=n_clientes, p=[0.4,0.3,0.1,0.1,0.1])

# ---------------------------
# Variables numéricas
# ---------------------------
frecuencia_compra = np.random.poisson(lam=5, size=n_clientes)  # mayoría baja, algunos altos
frecuencia_compra = np.clip(frecuencia_compra, 0, 60)

ticket_promedio_usd = np.random.lognormal(mean=3, sigma=0.5, size=n_clientes)  # 5–400 aprox
ticket_promedio_usd = np.clip(ticket_promedio_usd, 5, 400)

monto_total_usd = ticket_promedio_usd * (frecuencia_compra + np.random.randint(1, 5, size=n_clientes))
monto_total_usd = np.clip(monto_total_usd, 10, 10000)

num_categorias = np.random.randint(1, 21, size=n_clientes)

devoluciones_pct = np.clip(np.random.normal(loc=7, scale=5, size=n_clientes), 0, 40)
cupon_uso_pct = np.clip(np.random.normal(loc=20, scale=10, size=n_clientes), 0, 70)

# NPS con valores nulos (~10%)
nps = np.random.randint(-100, 101, size=n_clientes).astype(float)
mask_nulos = np.random.rand(n_clientes) < 0.1
nps[mask_nulos] = np.nan

# ---------------------------
# Variables de ingeniería
# ---------------------------
intensidad = monto_total_usd / (recencia_dias + 1)
lealtad = np.divide(num_categorias, frecuencia_compra + 1)  # evitar div/0
cliente_activo = (recencia_dias <= 90).astype(int)

# ---------------------------
# DataFrame final
# ---------------------------
df = pd.DataFrame({
    "id_cliente": ids,
    "fecha_primera_compra": primeras_compras,
    "fecha_ultima_compra": ultimas_compras,
    "pais": paises,
    "canal": canales,
    "frecuencia_compra": frecuencia_compra,
    "recencia_dias": recencia_dias,
    "ticket_promedio_usd": ticket_promedio_usd.round(2),
    "monto_total_usd": monto_total_usd.round(2),
    "num_categorias": num_categorias,
    "devoluciones_pct": devoluciones_pct.round(2),
    "cupon_uso_pct": cupon_uso_pct.round(2),
    "nps": nps,
    "rfm_recency": recencia_dias,
    "rfm_frequency": frecuencia_compra,
    "rfm_monetary": monto_total_usd.round(2),
    "aov": ticket_promedio_usd.round(2),
    "intensidad": intensidad.round(2),
    "lealtad": lealtad.round(2),
    "cliente_activo": cliente_activo
})

# Asegurar consistencia con frecuencia_compra
df['num_categorias'] = df['num_categorias'].astype(int)
df['num_categorias'] = np.where(
    df['frecuencia_compra'] == 0,
    0,
    np.minimum(df['num_categorias'].clip(lower=1), df['frecuencia_compra'])
).astype(int)

#recalcula lealtad sin artificios de +1:
df['lealtad'] = np.where(
    df['frecuencia_compra'] > 0,
    df['num_categorias'] / df['frecuencia_compra'],
    0.0
)

# Vista previa
print(df.head())

#Guardar la data como CSV 
output_csv = os.path.join(BASE_DIR, 'data_sintetica', 'clientes_sinteticos.csv')
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df.to_csv(output_csv, index=False)


#Guardar como pkl en data sintetica
output_pkl = os.path.join(BASE_DIR,'data_sintetica', 'clientes_sinteticos.pkl')
os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
df.to_pickle(output_pkl)