import pandas as pd
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import StandardScaler

# Übung-4
# Aufgabe 1 (Datensatz in Pandas DataFrame laden)
print('#'*150)
print('Aufgabe 1.')

path = "adult 2.csv"
data = pd.read_csv(path)

# Aufgabe 2 (die ersten Zeilen ausgeben)
print('#'*150)
print('Aufgabe 2.')

print(data.head())

# Aufgabe 3 (alle Spalten mit Zeichenketten in numerische Werte transformieren)
print('#'*150)
print('Aufgabe 3.')

cols_to_transform = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income']
data[cols_to_transform] = data[cols_to_transform].astype('category')
data[cols_to_transform] = data[cols_to_transform].apply(lambda x: x.cat.codes)
print(data)

# Aufgabe 4 (optimale Gruppenzahl ermitteln, hier = 4)
print('#'*150)
print('Aufgabe 4.')

# Wenn ergebnis von Scaler eine Parabel ist, Scaler nicht verwneden(hier kann keine exakte Gruppenanzahl ermittlet werden)
# einfach ausprobieren!

#s_scaler = StandardScaler()
#data = pd.DataFrame(s_scaler.fit_transform(data), columns=data.columns)
#print(data)

model = KMeans()
visualizer = KElbowVisualizer(model, K=(2, 9), timings=False)
visualizer.fit(data)
visualizer.show()

# Aufgabe 5 (Datensatz gruppieren)
print('#'*150)
print('Aufgabe 5.')

KMeans = KMeans(n_clusters=4)

# Aufgabe 6 (Tabelle mit neuer Spalte 'Label' (Gruppennummer) in neue CSV speichern)
print('#'*150)
print('Aufgabe 6.')

pred = KMeans.fit_predict(data)
data_new = pd.concat([data, pd.DataFrame(pred, columns=["label"])], axis=1)
print(data_new)
data_new.to_csv("./data_new_adult.csv")


	1. Laden Sie die Datei Adults.csv (mit Spaltennamen)
	2. Geben Sie die ersten Zeilen aus.
	3. Transformieren Sie alle Spalten mit Zeichenketten in numerische Werte.
	4. Ermitteln Sie die optimale Gruppenanzahl.
	5. Gruppieren Sie den Datensatz.
	6. Speichern Sie die Tabelle mit der neuen Spalte "Label" (Gruppennummer) in einer neuen CSV-Datei.

Hinweis zur Transformation:


cols_to_transform = ['workclass','education','marital-status','occupation','relationship','race','gender', 'native-country', 'income']
data[cols_to_transform] = data[cols_to_transform].astype('category')
data[cols_to_transform] = data[cols_to_transform].apply(lambda x: x.cat.codes)
print(data)