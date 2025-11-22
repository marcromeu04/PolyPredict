# Configuración de APIs para Datos Reales

## ¿Por qué necesito configurar APIs?

PolyPredict **solo funciona con datos reales** de Polymarket. Para obtener el máximo valor del sistema, necesitas configurar las siguientes APIs:

## APIs Requeridas vs Opcionales

### ✅ **Funciona SIN configuración:**
- **Polymarket API** - No requiere API key para funcionalidad básica
  - Obtiene mercados activos
  - Descarga historial de trades
  - Accede a orderbook

### 🔑 **Opcional pero RECOMENDADO:**

#### 1. **The Graph API** (Gratis - 100k queries/mes)

**¿Por qué?** Datos on-chain completos e históricos

**Cómo obtener:**
1. Ve a https://thegraph.com/studio/
2. Crea una cuenta
3. Copia tu API key
4. Agrégala a `.env`:
   ```bash
   THEGRAPH_API_KEY=tu_api_key_aqui
   ```

**Beneficios:**
- Historial completo de trades
- Datos de liquidez
- Análisis de top traders
- Verificación on-chain

---

#### 2. **PolygonScan API** (Gratis)

**¿Por qué?** Verifica datos directamente en blockchain

**Cómo obtener:**
1. Ve a https://polygonscan.com/apis
2. Crea cuenta gratis
3. Genera API key
4. Agrégala a `.env`:
   ```bash
   POLYGONSCAN_API_KEY=tu_api_key_aqui
   ```

**Beneficios:**
- Validación de transacciones
- Análisis de gas usage
- Historial de wallets
- Detección de patrones on-chain

---

#### 3. **NewsAPI** (Gratis - 100 requests/día)

**¿Por qué?** Detecta insider trading antes de anuncios

**Cómo obtener:**
1. Ve a https://newsapi.org/
2. Regístrate gratis
3. Copia tu API key
4. Agrégala a `.env`:
   ```bash
   NEWSAPI_KEY=tu_api_key_aqui
   ```

**Beneficios:**
- Correlación de trades con noticias
- Detección de timing sospechoso
- Análisis de eventos mayores

---

#### 4. **Twitter/X API** (Opcional)

**¿Por qué?** Sentiment analysis y early signals

**Cómo obtener:**
1. Ve a https://developer.twitter.com/
2. Aplica para developer account
3. Crea app y obtén Bearer Token
4. Agrégala a `.env`:
   ```bash
   TWITTER_BEARER_TOKEN=tu_token_aqui
   ```

**Beneficios:**
- Social media monitoring
- Early event detection
- Sentiment analysis

---

## Guía de Configuración Paso a Paso

### 1. Copia el archivo de ejemplo

```bash
cd /home/user/PolyPredict
cp .env.example .env
```

### 2. Edita el archivo `.env`

```bash
nano .env
# o usa tu editor favorito
```

### 3. Agrega tus API keys

```bash
# OBLIGATORIO (pero funciona sin key para básico)
# Polymarket ya está configurado por defecto

# RECOMENDADO
THEGRAPH_API_KEY=abcd1234...     # De thegraph.com/studio
POLYGONSCAN_API_KEY=XYZ789...    # De polygonscan.com/apis

# OPCIONAL
NEWSAPI_KEY=news123...           # De newsapi.org
TWITTER_BEARER_TOKEN=AAA...      # De developer.twitter.com
```

### 4. Guarda y cierra

```bash
# Ctrl+X, luego Y, luego Enter (en nano)
```

---

## Verificar Configuración

Ejecuta el test de datos reales:

```bash
python test_real_data.py
```

El test te dirá:
- ✅ Qué APIs están configuradas
- ⚠️ Cuáles faltan (pero son opcionales)
- ❌ Si hay problemas de conexión

---

## Comenzar a Usar

### Opción 1: Script principal

```bash
# Analizar el mercado con mayor volumen
python run_tracker.py

# Analizar un mercado específico
python run_tracker.py --market-id <market_id>

# Obtener más días de historial
python run_tracker.py --days 14
```

### Opción 2: Jupyter Notebook (Recomendado)

```bash
jupyter notebook notebooks/insider_tracker_demo.ipynb
```

El notebook incluye:
- Visualizaciones interactivas
- Análisis paso a paso
- Dashboard completo
- Exportación de resultados

---

## Límites de Rate

| API | Tier Gratis | Límite |
|-----|-------------|--------|
| Polymarket | Básico | ~1000 requests/hora |
| The Graph | Gratis | 100,000 queries/mes |
| PolygonScan | Gratis | 5 requests/segundo |
| NewsAPI | Gratis | 100 requests/día |

**Consejo:** El sistema tiene rate limiting automático para evitar exceder estos límites.

---

## Troubleshooting

### "No markets found"
- ✅ Verifica tu conexión a internet
- ✅ La API de Polymarket puede estar temporalmente down
- ✅ Intenta de nuevo en unos minutos

### "No trading data found"
- ✅ Algunos mercados nuevos no tienen trades aún
- ✅ Deja que el sistema auto-seleccione: `python run_tracker.py`
- ✅ O escoge un mercado con alto volumen

### "API key invalid"
- ✅ Verifica que copiaste la key completa
- ✅ No agregues espacios antes/después
- ✅ Asegúrate de estar en el archivo `.env` correcto

### "Rate limit exceeded"
- ✅ Espera unos minutos
- ✅ Reduce el parámetro `--days`
- ✅ El sistema esperará automáticamente

---

## Ejemplo de Uso Completo

```bash
# 1. Configurar
cp .env.example .env
nano .env  # Agregar API keys

# 2. Probar
python test_real_data.py

# 3. Analizar
python run_tracker.py

# 4. Ver resultados
ls data/results/
cat data/results/detection_report_*.json
```

---

## ¿Necesito TODAS las APIs?

**NO**. El sistema funciona con solo Polymarket API (que no requiere key).

**Pero:**
- Con **The Graph**: +100% más datos históricos
- Con **PolygonScan**: +Verificación on-chain
- Con **NewsAPI**: +Detección de timing con eventos

**Recomendación:**
1. Empieza solo con Polymarket (gratis, no requiere setup)
2. Agrega The Graph (5 minutos, gratis, gran mejora)
3. Agrega PolygonScan si quieres validación blockchain
4. Agrega NewsAPI si analizas mercados de eventos políticos/sociales

---

## Soporte

Si tienes problemas:
1. Revisa los logs en `logs/polypredict.log`
2. Ejecuta `python test_real_data.py` para diagnosticar
3. Verifica que `.env` existe y tiene las keys correctas
4. Revisa la documentación en `docs/API_INTEGRATIONS.md`

---

## Privacidad y Seguridad

- ⚠️ **NUNCA** compartas tu archivo `.env`
- ⚠️ **NUNCA** hagas commit de `.env` a git (ya está en .gitignore)
- ✅ Las API keys son solo para lectura de datos públicos
- ✅ No se requieren wallets ni private keys para detección
