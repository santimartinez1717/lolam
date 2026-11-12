# lolam

**¿Saber más o saber parar?** Cuánta de la mejora que dan el RL, la edición de pesos y otros arreglos a los modelos de razonamiento pequeños viene de que aprenden a **parar** a tiempo, y cuánta de que **saben más**.

El caso de partida es Qwen3.5-0.8B. En modo thinking saca un 7,9 % en MATH-500, y el 93 % de sus respuestas se corta por el límite de tokens. Con el thinking apagado saca ~52–55 %. Una edición de 131.072 pesos ([weight-space programming](https://rikiyat.github.io/blog/weight-space-programming/)) lo sube al 48 %.

Plan completo, hipótesis y criterios de éxito: [PLAN.md](PLAN.md).

## Estado

- [x] Fase 0: evaluación reanudable (vLLM), corrector de respuestas, pass@k, diff de pesos, Slurm
- [ ] Fase 0: líneas base en el cluster
- [ ] Fase 1: curvas A(t) de precisión con respuesta forzada
- [ ] Fase 2: GRPO
- [ ] Fase 3: stop-tuning
- [ ] Fase 4: análisis y geometría

## Uso

En local (Mac o CPU), sin GPU:

```bash
uv sync
uv run pytest
```

En el cluster:

```bash
bash scripts/setup_cluster.sh      # entorno + descarga de modelos y datos
bash scripts/phase0.sh smoke       # prueba de ~10 min
bash scripts/phase0.sh throughput  # rendimiento real de la GPU
bash scripts/phase0.sh diff        # qué cambió exactamente la edición WSP
bash scripts/phase0.sh baselines   # base con thinking / sin thinking / WSP
```

Para comparar las ejecuciones: `lolam-summary runs/*/math500 runs/*/gsm8k`.

Las evaluaciones se pueden reanudar. Los jobs de Slurm tienen un límite de 8 h, así que `slurm/chain.sh` encadena varios y cada uno continúa donde se quedó el anterior.
