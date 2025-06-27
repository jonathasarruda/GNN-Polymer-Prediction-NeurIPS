# GNN-Polymer-Prediction-NeurIPS
README: Análise e Previsão de Polímeros (MVP)
Este projeto foca na preparação de dados e modelagem para prever propriedades de polímeros.

🇧🇷 Português
Este repositório contém scripts para EDA (Análise Exploratória de Dados), pré-processamento e engenharia de features para dados de polímeros. O pipeline inclui a extração de features SMILES, tratamento de valores ausentes (com média) e normalização (StandardScaler). Um modelo de Rede Neural Gráfica (GNN) customizado, baseado em similaridade de cosseno para criar o grafo, é treinado para prever propriedades como Tg, FFV, Tc, Densidade e Rg. A performance é avaliada por wMAE (Weighted Mean Absolute Error). O processo é automatizado para reprodutibilidade.

🇪🇸 Español
Este repositorio contiene scripts para EDA (Análisis Exploratorio de Datos), preprocesamiento e ingeniería de características para datos de polímeros. El pipeline incluye la extracción de características SMILES, el manejo de valores faltantes (con la media) y la normalización (StandardScaler). Se entrena un modelo de Red Neuronal Gráfica (GNN) personalizado, basado en la similitud del coseno para construir el grafo, para predecir propiedades como Tg, FFV, Tc, Densidad y Rg. El rendimiento se evalúa mediante wMAE (Error Absoluto Medio Ponderado). El proceso está automatizado para la reproducibilidad.

🇷🇺 Русский
Этот репозиторий содержит скрипты для EDA (Разведочного Анализа Данных), предварительной обработки и генерации признаков для данных полимеров. Конвейер включает извлечение SMILES-признаков, обработку пропущенных значений (средним значением) и нормализацию (StandardScaler). Обучается настраиваемая модель Графовой Нейронной Сети (GNN), основанная на косинусном сходстве для построения графа, для предсказания таких свойств, как Tg, FFV, Tc, Плотность и Rg. Производительность оценивается с помощью wMAE (Взвешенной Средней Абсолютной Ошибки). Процесс автоматизирован для воспроизводимости.

🇬🇧 English
This repository contains scripts for EDA (Exploratory Data Analysis), preprocessing, and feature engineering for polymer data. The pipeline includes SMILES feature extraction, handling missing values (with mean imputation), and normalization (StandardScaler). A custom Graph Neural Network (GNN) model, based on cosine similarity for graph construction, is trained to predict properties such as Tg, FFV, Tc, Density, and Rg. Performance is evaluated using wMAE (Weighted Mean Absolute Error). The process is automated for reproducibility.
