import pandas as pd

ipc = pd.read_csv("data/ipc_sections.csv", encoding="latin1")
crpc = pd.read_csv("data/crpc_sections.csv", encoding="latin1")

print("IPC columns:", ipc.columns)
print("CRPC columns:", crpc.columns)
print(ipc.head())
print(crpc.head())