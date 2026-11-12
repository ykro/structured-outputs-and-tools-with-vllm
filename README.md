# Structured Outputs y Tool Calling con vLLM y Gemma

- **Por qué es relevante:** una app no lee prosa, lee campos. Si el modelo devuelve el JSON envuelto en Markdown, con una frase amable al inicio o con una urgencia que tu código no conoce, el parser truena en producción. vLLM puede garantizar la forma de la respuesta desde el servidor: convierte un JSON Schema, una lista de opciones o una expresión regular en una gramática y, en cada paso del decode, enmascara los tokens que la romperían. Con tool calling, el modelo decide qué función llamar y tu código decide si la ejecuta.
- **Para quién es:** desarrolladores que saben Python, ya llamaron alguna API de IA y ahora sirven su propio modelo (o vienen de [Tu primer servidor vLLM con Gemma](https://github.com/ykro/first-vllm-server-with-gemma)). No hace falta JAX ni PyTorch.
- **Quién debería verlo:** quien construye una integración donde la salida del modelo alimenta a otro sistema: tickets de soporte que se vuelven registros, formularios que se llenan solos, agentes que consultan una base de datos o crean una orden.
- **Serie completa:** cuarta unidad del [vLLM for Software Devs Field Guide](https://github.com/ykro/vllm-for-software-devs-field-guide). La pregunta de esta unidad es «¿Cómo confío en lo que devuelve?».

---

## La app que atraviesa los cinco experimentos

Una tienda en línea de Guatemala recibe tickets de soporte en texto libre («Me llegó la licuadora con el vaso rajado, pedido GT-220871. Quiero que me devuelvan mi dinero.»). La app necesita convertir cada ticket en un registro con cinco campos:

| Campo | Valores permitidos |
| :--- | :--- |
| `category` | `shipping`, `damaged_product`, `wrong_item`, `payment`, `account`, `cancellation`, `other` |
| `urgency` | `low`, `medium`, `high` |
| `product` | texto o `null` |
| `requested_action` | `refund`, `exchange`, `resend`, `tracking`, `cancellation`, `information`, `invoice` |
| `order_id` | `GT-` seguido de 6 dígitos, o `null` |

Los 20 tickets fijos y el registro que escribiría una persona para cada uno están en `data/tickets.json`. El esquema y un validador pequeño escrito solo con la biblioteca estándar están en `vllm_lab/schema.py`; el mismo validador juzga el texto libre y la salida restringida, así que las dos se miden con la misma regla.

---

## Modelo mental

### 1. Pedir JSON en el prompt es un deseo

El experimento 01 manda los 20 tickets con un system prompt que termina en «Responde solo con el JSON» y pasa cada respuesta por lo que haría la app: `json.loads` y luego el validador. Las fallas típicas son tres: el JSON llega dentro de un bloque de Markdown, llega con una frase antes, o se puede leer pero trae un valor fuera del esquema (`"urgency": "urgente"`, un `order_id` sin el prefijo `GT-`, un campo que falta). Las dos primeras rompen el parser; la tercera es peor, porque pasa el parser y rompe algo más adelante.

### 2. El esquema va en la petición

El experimento 02 manda los mismos tickets con `response_format={"type": "json_schema", "json_schema": {...}}`. vLLM compila el esquema en una gramática y, durante el decode, solo deja pasar tokens que mantienen el texto dentro de ella. El resultado siempre se puede parsear y siempre valida. El trade-off es latencia: la primera petición con un esquema nuevo paga la compilación de la gramática (después queda en caché) y cada paso del decode hace un poco más de trabajo. El experimento mide las dos cosas contra el texto libre.

La gramática garantiza la forma. Que la categoría sea la correcta es otra pregunta, y por eso el juez también reporta, solo como información, cuántos campos coinciden con las etiquetas escritas a mano.

### 3. Contratos más pequeños, `choice` y `regex`

Cuando la app necesita un solo valor, un esquema completo sobra. El experimento 03 pide la urgencia con `choice` (una de tres palabras) y el número de pedido con `regex` (`GT-[0-9]{6}|NONE`). La alternativa `NONE` importa: si la expresión solo aceptara `GT-` y seis dígitos, el modelo estaría obligado a inventar un número para los tickets que no traen ninguno.

En vLLM 0.33 estas restricciones van en un solo campo de la petición, `structured_outputs`, con exactamente una de estas llaves: `json`, `regex`, `choice`, `grammar` o `json_object`. Con el SDK de OpenAI se mandan en `extra_body`:

```python
client.chat.completions.create(
    model="gemma",
    messages=msgs,
    extra_body={"structured_outputs": {"choice": ["low", "medium", "high"]}},
)
```

Las versiones anteriores al cambio de nombre usaban `guided_choice` y `guided_regex` como campos sueltos. `vllm_lab/client.py` intenta primero `structured_outputs` y, si el servidor responde 400, repite con los nombres anteriores; el juez imprime cuál de los dos funcionó.

### 4. Tool calling, el modelo decide y tu código actúa

El experimento 04 define dos herramientas en Python con datos falsos (`vllm_lab/store.py`): `lookup_order(order_id)` y `create_return(order_id, reason)`. El modelo solo ve sus nombres, descripciones y parámetros por medio de la API `tools` de OpenAI. El loop manda la conversación, ejecuta las llamadas que devuelve el modelo, agrega cada resultado como mensaje `tool` y repite hasta que el modelo responde en texto. Las reglas de negocio (30 días para devolver) viven en el código de la herramienta, así que una llamada indebida devuelve un error en lugar de crear la devolución.

Cuatro mensajes fijos revisan el comportamiento: una consulta de estado llama `lookup_order`, una devolución válida termina en `create_return` con el número de pedido correcto, una devolución fuera de plazo consulta el pedido y no crea nada, y una pregunta general no llama ninguna herramienta.

**Flags del servidor.** vLLM solo devuelve `tool_calls` si arranca con tool calling activado. Para Gemma 4 en vLLM 0.33:

```text
--enable-auto-tool-choice --tool-call-parser gemma4
```

`gcp/deploy_vllm.sh` ya los pasa. El parser `gemma4` está registrado en `vllm/tool_parsers/__init__.py` de la versión 0.33.0 (clase `Gemma4EngineToolParser`) y traduce los tokens nativos de llamada de Gemma 4 al formato `tool_calls` de OpenAI. El experimento usa `tool_choice="auto"`, así el modelo decide solo cuándo llamar; `required` y una herramienta con nombre no se probaron en esta corrida. Si el template del modelo no produce llamadas, la imagen trae uno alternativo en `/vllm-workspace/examples/tool_chat_template_gemma4.jinja`, que se activa con `VLLM_CHAT_TEMPLATE`.

### 5. Streaming de un JSON hacia la interfaz

El experimento 05 hace streaming de un registro con `response_format` y, después de cada chunk, intenta leer el texto recibido hasta ese momento (`parse_partial` en `vllm_lab/schema.py`). Los campos cuyo valor ya cerró se pueden mostrar con confianza; el campo que se está escribiendo se puede mostrar como «escribiendo». Así la interfaz enseña la categoría y la urgencia antes de que termine el decode, y al final el texto completo se valida igual que una respuesta sin streaming.

---

## Resultados de verificación (`verify_structured.py`)

Corrida en vivo del 2026-11-11 con `make verify-live` (vLLM 0.33.0, `google/gemma-4-E4B-it` en bf16, una NVIDIA L4 de Cloud Run en `us-east4`, pesos desde un bucket, servidor con `--enable-auto-tool-choice --tool-call-parser gemma4`). Todo queda en `benchmarks/latest_results.json` y `benchmarks/terminal_output.txt`. El juez pasa los 5 checks.

| Experimento | Qué se mide | Resultado en L4 |
| :--- | :--- | :--- |
| **01. JSON pedido en el prompt** | Cuántas respuestas parsean y validan | **0 de 20** parsean: el modelo envuelve cada JSON en un bloque de Markdown. Al quitar los fences y la charla, **19 de 20** validan; el que falla trae `requested_action = "return"`, un valor fuera del esquema |
| **02. `response_format` con JSON Schema** | Validación y latencia contra texto libre | **20 de 20** parsean y validan. Latencia media 2.38 s contra 2.67 s del texto libre (0.89 veces): 54 tokens por registro contra 62, porque no hay fences |
| **03. `choice` y `regex`** | Urgencia en las 3 opciones y número de pedido con patrón | 100 % y 100 %. El texto libre también acertó la forma en las dos (100 % y 100 %): con una pregunta de una palabra este modelo ya obedecía, y la restricción convierte esa costumbre en garantía |
| **04. Loop de tool calling** | Qué herramientas llama en 4 casos | **4 de 4**: `lookup_order`, `lookup_order` + `create_return`, `lookup_order` sin devolución (47 días), ninguna herramienta |
| **05. Streaming de JSON a la interfaz** | Cuándo llega cada campo | Primer token a los 0.31 s, `category` cerrada a los 0.69 s, `urgency` 1.03, `product` 1.39, `requested_action` 1.77, `order_id` 2.43 y fin a los 2.51 s (58 chunks). El texto completo valida |

Salida del juez en vivo:

```text
Experiment                       | Measured                           | Result                     | Check
----------------------------------------------------------------------------------------------------------
01. JSON asked in the prompt     | parse 0/20, valid 0/20             | unwrapped 19/20 valid      | PASS
02. response_format json_schema  | valid 100 % vs 0 % free            | latency 0.89x free         | PASS
03. choice and regex             | enum 100 %, regex 100 %            | free 100 % / 100 %         | PASS
04. Tool calling loop            | lookup, lookup+return, lookup, -   | 4/4 cases                  | PASS
05. Streaming JSON to a UI       | first field 0.693 s of 2.507 s     | parses at end              | PASS
----------------------------------------------------------------------------------------------------------
Agreement with hand labels (informative): schema 92 % of fields, urgency 80 %, order id 100 %.
```

### Lo que falló primero

La primera corrida en vivo del experimento 04 sacó 3 de 4. Con un system prompt corto («Usa las herramientas...»), el modelo respondió a «¿por dónde viene mi pedido GT-104233?» con una pregunta de aclaración y no llamó `lookup_order`. Se refuerza el prompt (consultar siempre el pedido cuando haya número, sin pedir aclaraciones) y pasan los 4. Tool calling sigue siendo una decisión del modelo; por eso el juez revisa casos fijos. La primera corrida queda en `benchmarks/tool_calling_first_attempt.json`.

### Cómo leer estas cifras

- **Comportamiento contra números.** El juez revisa comportamiento: el esquema siempre valida, las opciones y el patrón siempre se respetan, el loop llama las herramientas correctas y el streaming termina en un JSON válido. La tasa del texto libre y las latencias cambian con el modelo, la versión y la corrida; se reportan y no se juzgan.
- **Latencia del esquema.** La primera petición con un esquema nuevo (máximo 60 tokens) tardó 1.57 s, y la compilación de la gramática no se distingue del ruido del túnel. El cociente `latency x free` compara las medias de los 20 tickets: 0.89, porque el texto libre genera 8 tokens más por registro (fences de Markdown) y eso pesa más que el costo de la máscara.
- **Texto libre contra texto limpio.** El juez cuenta el texto libre tal cual, como lo recibe un `json.loads` ingenuo (0 de 20), y aparte después de quitar los fences y la charla (19 de 20, columna `unwrapped`). La diferencia entre las dos cifras es lo que cuesta no tener un contrato.
- **Forma contra contenido.** La línea de coincidencia con las etiquetas escritas a mano es informativa: con el esquema, 92 % de los campos coinciden y la urgencia con `choice` coincide en 16 de 20 tickets (80 %); los 4 que no, el modelo los clasificó con una urgencia más alta o más baja que la etiqueta humana. Un registro puede validar y aun así traer la categoría equivocada; para eso hace falta una evaluación aparte, con datos etiquetados como los de `data/tickets.json`.
- **El túnel.** Los tiempos en vivo incluyen la ida y vuelta por `gcloud run services proxy`.

---

## Estructura del repositorio

```text
.
├── benchmarks/
│   ├── latest_results.json                # lo escribe make verify-live
│   ├── terminal_output.txt                # salida del juez en vivo
│   └── tool_calling_first_attempt.json    # el caso que falló antes de reforzar el prompt
├── data/
│   └── tickets.json                       # 20 tickets en español y su registro esperado
├── diagrams/                              # 3 diagramas blue-gold (html, svg, png)
├── experiments/
│   ├── 01_free_text_breaks_parsers.py     # JSON pedido solo en el prompt: qué parsea y qué valida
│   ├── 02_json_schema_output.py           # response_format json_schema: validación y latencia
│   ├── 03_enums_and_regex.py              # structured_outputs con choice y regex
│   ├── 04_tool_calling.py                 # Loop de tools con lookup_order y create_return
│   └── 05_streaming_json_to_ui.py         # Streaming de un registro y campos parciales
├── gcp/
│   ├── deploy_vllm.sh                     # Cloud Run + L4 on-demand con tool calling activado
│   ├── stage_weights.sh                   # Copia los pesos a Cloud Storage una vez
│   └── teardown.sh                        # Borra el servicio
├── tests/
│   └── test_structured.py                 # Validador, parser parcial, tools y los 5 experimentos
├── vllm_lab/
│   ├── cli.py                             # --mock, --base-url y salida en pantalla
│   ├── client.py                          # Cliente OpenAI apuntando a vLLM, constraints con fallback
│   ├── extract.py                         # Corre los 20 tickets y los juzga
│   ├── mock_server.py                     # Servidor falso determinista (sin GPU)
│   ├── schema.py                          # JSON Schema, validador y parser de JSON parcial
│   ├── store.py                           # Herramientas en Python con datos falsos
│   └── tickets.py                         # Carga de tickets y prompts
├── Makefile
├── pyproject.toml
└── verify_structured.py                   # El Outside Judge
```

---

## Paso a paso

### 1. Instalación

```bash
git clone https://github.com/ykro/structured-outputs-and-tools-with-vllm.git
cd structured-outputs-and-tools-with-vllm
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

La única dependencia es el SDK de `openai`; el validador, el parser parcial, el servidor falso y las herramientas usan solo la biblioteca estándar.

### 2. Sin GPU

```bash
# ruff + pytest + el juez contra el servidor falso
make all

# Cada experimento por separado contra el servidor falso
python experiments/01_free_text_breaks_parsers.py --mock
python experiments/02_json_schema_output.py --mock
python experiments/03_enums_and_regex.py --mock
python experiments/04_tool_calling.py --mock
python experiments/05_streaming_json_to_ui.py --mock
```

### 3. Con vLLM en Cloud Run

vLLM necesita una GPU NVIDIA. El script despliega la imagen oficial `vllm/vllm-openai:v0.33.0` con Gemma 4 en Cloud Run con una L4 on-demand (`min-instances 0`), sin URL pública.

```bash
export GCP_PROJECT_ID=<tu-proyecto>
bash gcp/stage_weights.sh     # opcional, una vez: pesos en Cloud Storage para cold starts más rápidos
make deploy                   # (MIN_INSTANCES=1 make deploy mantiene la GPU caliente mientras mides; make cool la devuelve a 0) incluye --enable-auto-tool-choice --tool-call-parser gemma4

# En otra terminal, el túnel local (queda escuchando)
gcloud run services proxy vllm-gemma --region us-east4 --port 8000

# Las cifras reales: escribe benchmarks/latest_results.json y benchmarks/terminal_output.txt
make verify-live

# Lo mismo, imprimiendo además el resultado completo de cada experimento
make report

# Al terminar
make teardown
```

Si ya desplegaste el servicio de `first-vllm-server-with-gemma` sin tool calling, vuelve a correr `make deploy`: el servicio se actualiza con los flags nuevos. La región por defecto es `us-east4` y se cambia con `GCP_REGION`.
