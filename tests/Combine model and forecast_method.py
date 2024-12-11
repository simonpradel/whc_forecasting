# Databricks notebook source
import itertools

model = ["AutoARIMA", "AutoETS", "AutoTS", "AutoGluon"]
forecast_method = ["global", "level"]
noLevelForStatModel = True
time_limit = [600,1800]

def ensure_iterable(value):
    if value is None:
        return [None]
    elif isinstance(value, (list, tuple)):
        return value
    else:
        return [value]
            
model = ensure_iterable(model)
forecast_method = ensure_iterable(forecast_method)

# Erzeuge alle Kombinationen
configurations = list(itertools.product(model, forecast_method, time_limit))

# Filtere ungewollte Kombinationen, falls `noLevelForStatModel == True`
if noLevelForStatModel:
    configurations = [
        (m, f, t) for m, f, t in configurations 
        if not (f == "level" and m in ["AutoETS", "AutoARIMA"])
    ]

configurations
