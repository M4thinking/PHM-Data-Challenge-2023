# Beta-VAE para Clasificación de Datos de Acelerómetro en Caja de Cambios

Este proyecto se centra en la implementación de un beta-VAE para clasificar datos de un acelerómetro montado en la superficie de una caja de cambios. El objetivo principal es resolver un problema de clasificación con puntos de operación y clases faltantes.

## Descripción

El proyecto utiliza técnicas de aprendizaje no supervisado y semi-supervisado para abordar el problema de clasificación. Se propone un modelo que actúa simultáneamente como clasificador y regresor. Cada uno de estos submodelos propone la etiqueta a comprar por el otro cuando la etiqueta en cuestión no existe.

## Características

- Implementación de un beta-VAE para clasificación de datos de acelerómetro.
- Uso de técnicas de aprendizaje no supervisado y semi-supervisado.
- Propuesta de un modelo simultáneo de clasificación y regresión.
- Manejo de situaciones con puntos de operación y clases faltantes.
- Uso de botorch para optimización bayesiana de la arquitectura VAE.