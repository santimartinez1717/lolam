# lolam — Plan de investigación (v2)

> **¿Saber más o saber parar?**
> Cuánta de la mejora que dan el RL, la edición de pesos y otros arreglos a los modelos de razonamiento pequeños viene de que **aprenden a parar**, y cuánta de que **saben más**.

La v1 de este plan giraba en torno a reproducir *weight-space programming* (WSP) y cruzar mapas por capa. La descarté tras revisar el estado del arte (§1.3). Este documento la sustituye.

---

## 0. Resumen

**El fenómeno.** En modo thinking, Qwen3.5-0.8B saca un 7,9 % en MATH-500, y el 93 % de sus respuestas se corta por el límite de tokens. Con el thinking apagado saca ~52–55 %. Una edición de pesos de 131.072 parámetros (WSP) lo sube al 48 %. GRPO también lo mejora.

**La pregunta.** Cuando un método "mejora el razonamiento" de un modelo pequeño, ¿le enseña algo o solo le enseña a **dejar de pensar a tiempo**?

**La herramienta.** La curva de **precisión en cualquier instante**, A(t). Se corta el razonamiento del modelo tras t tokens y se le obliga a responder (`</think>` seguido de `\boxed{`). Con una sola generación larga por muestra y respuestas forzadas cortas (que aprovechan la caché del prefijo), se ve qué sabía el modelo en cada momento del razonamiento.

- **Arreglo de parada:** la curva A(t) apenas cambia, pero el modelo cierra el razonamiento cerca del máximo de la curva.
- **Enseñanza:** la curva A(t) sube.

**El experimento constructivo.** Si la mayor parte es parada, un fine-tuning que solo enseña *cuándo* cerrar el razonamiento ("stop-tuning") debería recuperar la mayor parte de la ganancia de GRPO por una fracción del cómputo.

---

## 1. Por qué esta pregunta

### 1.1 Es general

Afecta a cualquier modelo de razonamiento pequeño y a cualquier método de post-entrenamiento. No depende del paper de WSP, aunque lo usa como caso de estudio.

### 1.2 Lo que ya se sabe y en qué se apoya

- **El RLVR mejora pass@1 pero no pass@k.** Afina capacidades que ya existen en lugar de crear nuevas (Yue et al. 2025; *RL vs. Distillation*, 2505.14216).
- **Los modelos de razonamiento entran en bucles,** sobre todo los pequeños y los destilados (*Why do reasoning models loop?*, PMLR v306).
- **Overthinking y early exit:** los estados ocultos codifican si la respuesta intermedia ya es correcta (2504.05419; NEAT, LYNX).
- **WSP:** la edición de Qwen quita los bucles. XLRA encontró que el entrenamiento posterior "desgasta" ese control de bucles.

### 1.3 Lo que no encontré hecho

Estas tres cosas faltan en lo que he revisado (hay que volver a comprobarlo antes de publicar):

- **Una descomposición sistemática** de la ganancia de distintos métodos en una parte de *parada* y una parte de *saber*, medida con A(t), sobre los mismos modelos y problemas.
- **Una prueba directa de que basta enseñar a parar:** stop-tuning frente a GRPO con el mismo modelo y los mismos datos.
- **La comparación en esos términos** de métodos muy distintos: RL, edición de pesos, steering, decodificación y modo sin thinking.

Si al revisar aparece alguno ya hecho, se ajusta el alcance. La herramienta y los resultados siguen teniendo valor como réplica.

---

## 2. Hipótesis

Los criterios se fijan antes de ver los datos.

**Definiciones:**
- **A_b(t):** precisión con respuesta forzada tras t tokens de razonamiento, para el modelo b.
- **Parada oráculo del modelo base:** S* = max_t A_base(t), es decir, la precisión que tendría el modelo base si parase siempre en el mejor presupuesto fijo.
- **Ganancia de un método M:** G = Acc(M) − Acc(base).
- **Parte de parada:** min(G, S* − Acc(base)). **Parte de saber:** el resto.

| # | Hipótesis | Se confirma si… |
|---|-----------|-----------------|
| **H1** | En Qwen3.5-0.8B la mayor parte de la ganancia de **GRPO** y de la **edición WSP** es parada. | La parte de parada es ≥ 70 % de G en MATH-500 y GSM8K, con intervalos bootstrap. |
| **H2** | El **stop-tuning** recupera la mayor parte de la ganancia de GRPO. | Recupera ≥ 70 % de G_GRPO con ≤ 20 % de su cómputo. |
| **H3** | Lo que el modelo "sabe" no cambia mucho. | La curva A(t) de GRPO y de WSP queda dentro de ±5 puntos de la del modelo base en el rango de t relevante. |
| **H4** (secundaria, geometría) | El momento de parar es **linealmente decodificable** del residual stream, y la dirección de bucles de WSP se relaciona con él. | Un probe lineal predice "la respuesta forzada ya es correcta y estable" con AUROC > 0,8. La similitud coseno con la dirección WSP es mayor que con direcciones aleatorias (p < 0,05). |

**Si fallan:** que H1 falle significa que GRPO sí enseña algo en un modelo de 0,8B. Es un resultado igual de interesante y se publica igual.

**Cuidado con un sesgo.** Dar la respuesta forzada puede infraestimar lo que el modelo sabe, porque cortar a mitad de una frase le perjudica. Para mitigarlo, se corta en el siguiente salto de línea después de t y se comparan dos plantillas de respuesta forzada.

---

## 3. Diseño

### 3.1 Modelos

- **Principal: Qwen3.5-0.8B,** con su edición WSP pública (`RikiyaT/Qwen3.5-0.8B-weight-space-programming`, que cambia solo las capas 15 y 19 de `down_proj`).
  - **Arquitectura:** 24 capas híbridas (atención lineal con atención completa cada 4), multimodal.
  - **Requisitos:** vLLM ≥ 0.33 y transformers ≥ 5.20.2.
- **Réplica si hay tiempo:** otro modelo pequeño con thinking (Qwen3.5-2B o Qwen3-1.7B) y checkpoints de RL públicos del mismo tamaño. Así se comprueba que no es algo exclusivo de un modelo.

### 3.2 Tareas

| Tarea | Para qué sirve |
|-------|----------------|
| **MATH-500**, subconjunto fijo de 200 | La principal, conecta con WSP |
| **GSM8K**, subconjunto fijo de 300 | Más fácil; aquí la parada debería dominar todavía más |
| Entrenamiento: MATH (train) y GSM8K (train), sin solapar con la evaluación | Para GRPO y el stop-tuning |

### 3.3 Condiciones

| Condición | Qué es | Cambia pesos |
|-----------|--------|--------------|
| Base, con thinking | La referencia | — |
| Base, sin thinking | Lo que el modelo hace sin pensar | — |
| Base con presupuesto forzado | Corte a presupuesto fijo t\* elegido en validación | — |
| WSP | La edición pública | Sí (0,016 %) |
| **GRPO** | LoRA y recompensa binaria de corrección, receta fija | Sí |
| **Stop-tuning** | SFT con LoRA sobre trazas propias del modelo base, cortadas en el primer punto donde la respuesta forzada es correcta y se mantiene, con `</think>` y la respuesta añadidos | Sí |
| (Opcional) Steering | Restar la dirección de bucles calculada por diferencia de medias | No (en inferencia) |

### 3.4 Métricas

- **Precisión** y **tasa de truncado** (si la generación acaba por el límite de longitud).
- **Longitud media** del razonamiento.
- **Curva A(t)** con t ∈ {256, 512, 1k, 2k, 4k, 8k, 16k}, y la descomposición del §2.
- **pass@k** para k ≤ n, para ver si cambia lo que el modelo puede llegar a resolver.
- **Cómputo** de cada método: horas de GPU y tokens generados.

---

## 4. Fases

### Fase 0 · Infraestructura (semanas 1–2)

- **Entorno:** `uv`, plantillas de Slurm y jobs encadenados para el límite de 8 h.
- **Evaluación reanudable:** se escribe el resultado de cada problema según termina, así un job que muere no pierde trabajo.
- **Corrección:** pass@k y corrector de respuestas, con tests.
- **Rendimiento:** medir los tokens por segundo reales de la porción de GPU.
- **Comprobar el modelo editado:** confirmar que solo difieren 2 tensores y extraer la dirección editada (SVD del cambio de pesos).
- **Línea base:** evaluar el modelo base con thinking y sin thinking, y el modelo WSP.
- **Entregable:** tabla de líneas base, que reproduce las cifras públicas con nuestro propio código.

### Fase 1 · Herramienta A(t) (semanas 2–3)

- **Respuestas forzadas** con la caché de prefijos de vLLM, y validación de las plantillas de respuesta forzada (ver el sesgo del §2).
- **Curvas A(t)** del modelo base y del modelo WSP, y cálculo de la parada oráculo S\*.
- **Entregables:** `lolam.anytime` (sirve para cualquier modelo de HF) y la primera figura del proyecto.

### Fase 2 · GRPO (semanas 3–6)

- **Configuración:** TRL GRPO, LoRA, vLLM colocado en la misma GPU, máximo de 4k–8k tokens por respuesta durante el entrenamiento y 3 semillas si el presupuesto lo permite.
- **Seguimiento:** checkpoints intermedios para ver cómo evoluciona la curva A(t) durante el entrenamiento.
- **Riesgo principal:** que entrenar la atención lineal de Qwen3.5 necesite kernels específicos (fla) y vaya lento. Se comprueba en la Fase 0. Plan B: RFT (SFT sobre soluciones correctas propias).

### Fase 3 · Stop-tuning (semanas 5–7)

- **Construir los datos** a partir de las trazas del modelo base que ya genera la Fase 1, así que no cuesta generación extra.
- **SFT con LoRA,** con un presupuesto mucho menor que el de GRPO.
- **Comprobar H2.**

### Fase 4 · Análisis y geometría (semanas 7–9)

- **H1 y H3:** la descomposición y la comparación de curvas.
- **H4:** probe lineal sobre el residual stream, comparado con la dirección WSP.
- **Robustez:** variar las plantillas de respuesta forzada, los subconjuntos y, si hay tiempo, el segundo modelo.

### Fase 5 · Escritura (semanas 9–10)

- **Informe técnico** en formato arXiv.
- **Post de blog** con las curvas A(t).
- **Publicación** del código, las trazas y los LoRAs.

---

## 5. Presupuesto de cómputo

Recursos: porción MIG de una H200 con 16 GB y ~1/7 de la potencia, 16 GB de RAM y jobs de 8 h. **Todas las cifras son estimaciones y se recalculan en la Fase 0.**

| Bloque | Estimación |
|--------|------------|
| Líneas base: 3 condiciones × 500 problemas × n = 4, hasta 32k tokens (el modelo base agota el límite casi siempre) | ~25–40 h. **Si no cabe**, n = 2 y un máximo de 16k, y se dice explícitamente |
| Curvas A(t): reutilizan las trazas, las respuestas forzadas son cortas | ~3–6 h |
| GRPO: 1–3 semillas (XLRA necesitó 23 h en una RTX 5090 para un caso parecido) | ~40–80 h |
| Stop-tuning | ~3–6 h |
| **Total** | **~75–130 h de GPU ≈ 10–17 jobs de 8 h** |

---

## 6. Riesgos

| Riesgo | Mitigación |
|--------|------------|
| La respuesta forzada infraestima lo que el modelo sabe | Cortar en saltos de línea, comparar 2 plantillas y validar con el modelo sin thinking |
| GRPO no cabe o va muy lento con la arquitectura híbrida | Contexto de entrenamiento más corto; plan B con RFT |
| Las generaciones de 32k hacen inviable evaluar el modelo base | Subconjunto fijo y n bajo; la curva A(t) hasta 16k basta para H1 |
| Alguien publica lo mismo | Se publica pronto la herramienta y los primeros resultados (Fase 1), y se sigue con H2 y H4 |
| El corrector de respuestas falla | Tests con casos difíciles y revisión manual de una muestra de 50 |

---

## 7. Entregables para GitHub y el CV

Los primeros salen aunque las hipótesis fallen:

1. **`lolam.anytime`:** herramienta reutilizable para medir qué sabe un modelo de razonamiento en cada momento.
2. **Tabla de líneas base** reproducible: thinking, sin thinking y WSP sobre Qwen3.5-0.8B.
3. **La respuesta a H1–H3,** con figuras claras, sea positiva o negativa.
4. **Stop-tuning** como receta barata, si funciona.
5. **Informe técnico, post de blog, trazas publicadas y `make reproduce`.**

---

## 8. Estructura del repositorio

```
lolam/
├── PLAN.md  README.md  pyproject.toml
├── lolam/
│   ├── eval/      # datos, corrector, pass@k, generación reanudable, resumen
│   ├── anytime/   # curvas A(t) con respuesta forzada        (Fase 1)
│   ├── rl/        # GRPO + RFT                               (Fase 2)
│   ├── stoptune/  # datos y SFT de parada                     (Fase 3)
│   ├── probe/     # probes y geometría                        (Fase 4)
│   ├── wsp/       # diff del modelo editado, dirección extraída
│   └── bench/     # rendimiento de la GPU
├── slurm/  configs/  tests/  reports/
```
