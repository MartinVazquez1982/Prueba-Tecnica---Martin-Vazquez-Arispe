# Prueba Técnica — Martin Vazquez Arispe

Asistente automatizado de soporte técnico que responde preguntas basándose en documentación interna. Utiliza Python (FastAPI + FAISS) para la ingesta y recuperación semántica, n8n como orquestador del flujo, y OpenAI GPT como modelo de lenguaje.

## Arquitectura

```
Usuario
  │
  ▼
n8n Webhook (POST /webhook/ask)
  │
  ├── valida input vacío → 400
  │
  ▼
Python API (POST /query)          ← búsqueda semántica sobre FAISS
  │
  ├── sin resultados → "No tengo información..."
  ├── API caída      → 503
  │
  ▼
OpenAI GPT-4.1-mini               ← genera respuesta con contexto recuperado
  │
  ▼
Respuesta JSON al usuario
```

## Requisitos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Clave de API de OpenAI ([obtener aquí](https://platform.openai.com/api-keys))

## Configuración

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/MartinVazquez1982/Prueba-Tecnica---Martin-Vazquez-Arispe.git
   cd Prueba-Tecnica---Martin-Vazquez-Arispe
   ```

2. Crear el archivo de variables de entorno:
   ```bash
   cp .env.example .env
   ```

3. Editar `.env` y completar los valores:
   ```env
   OPENAI_API_KEY=sk-...
   N8N_ENCRYPTION_KEY=cualquier-cadena-secreta-aleatoria
   ```

## Levantar el proyecto

```bash
docker compose up --build
```

Esto ejecuta tres servicios en orden:

| Servicio | Descripción |
|----------|-------------|
| `ingest` | Lee la carpeta `/docs`, genera embeddings y construye el índice FAISS. Se ejecuta una sola vez. |
| `api` | API REST de recuperación semántica. Disponible en `http://localhost:8000` |
| `n8n` | Motor de workflow. Disponible en `http://localhost:5678` |

> La primera vez puede tardar unos minutos mientras se descargan las imágenes y se generan los embeddings.

## Importar el workflow en n8n

1. Abrir `http://localhost:5678`
2. Ingresar con usuario `admin` y contraseña `admin`
3. En el menú superior izquierdo ir a **Workflows → Add Workflow**
4. Hacer click en el menú `⋮` → **Import from file**
5. Seleccionar el archivo `n8n/workflow.json`
6. Activar el workflow con el toggle **Inactive → Active** (esquina superior derecha)

### Configurar las credenciales de OpenAI

> **Importante:** n8n cifra las credenciales con la `N8N_ENCRYPTION_KEY` de quien exportó el workflow. Al importarlo en una instancia nueva, el nodo de OpenAI aparecerá sin credenciales y hay que configurarlas manualmente.

1. Hacer click en el nodo **"Generate Response"**
2. En el campo **Credential** hacer click en **Create new credential**
3. Ingresar el `OPENAI_API_KEY` del archivo `.env`
4. Guardar y cerrar

## Probar el asistente

Con el workflow activo, enviar preguntas al webhook:

```bash
curl -X POST http://localhost:5678/webhook/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Cómo reinicio el servicio de autenticación?"}'
```

Respuesta esperada:

```json
{
  "answer": "Para reiniciar el servicio de autenticación..."
}
```

### Ejemplos de preguntas

```bash
# Consulta sobre autenticación
curl -X POST http://localhost:5678/webhook/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Cómo me autentico en el sistema?"}'

# Consulta sobre un error
curl -X POST http://localhost:5678/webhook/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "El sistema devuelve error 502, ¿qué significa?"}'

# Pregunta sin información en la documentación
curl -X POST http://localhost:5678/webhook/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Cuál es el precio del plan premium?"}'
```

## Verificar que la API está funcionando

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "index_loaded": true,
  "total_chunks": 42
}
```

## Manejo de errores

| Caso | Respuesta |
|------|-----------|
| Pregunta vacía | `400` — `"La pregunta no puede estar vacía."` |
| Sin información en la documentación | `200` — `"No tengo información sobre eso en la documentación."` |
| API Python no disponible | `503` — `"El servicio de búsqueda no está disponible."` |
| Error de OpenAI (rate limit) | `429` — `"OpenAI rate limit reached. Try again later."` |
| Error de conexión con OpenAI | `502` — `"Could not connect to OpenAI API."` |

## Estructura del proyecto

```
.
├── docs/                   # Documentación fuente (input del sistema)
├── n8n/
│   └── workflow.json       # Workflow exportado listo para importar
├── python/
│   ├── readers/            # Parsers por tipo de archivo (.txt, .md, .pdf, .json)
│   ├── tests/              # Tests unitarios
│   ├── api.py              # FastAPI — endpoint /query y /health
│   ├── ingest.py           # Pipeline de ingesta y construcción del índice
│   ├── config.py           # Configuración centralizada
│   ├── schemas.py          # Modelos Pydantic
│   └── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

## Ejecutar los tests

```bash
cd python
pip install -r requirements.txt -r requirements-dev.txt
pytest tests/ -v
```
