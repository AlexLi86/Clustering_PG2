import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import StandardScaler

# Übung-3
# Übung 1.1 (Datensatz in Pandas DataFrame laden)
print('#'*150)
print('Aufgabe 1.1')
path = "adult 2.csv"
data = pd.read_csv(path)

# Übung 1.2 (die ersten und letzten 5 ausgeben)
print('#'*150)
print('Aufgabe 1.2')
print(data.head())
print(data.tail())

# Übung 1.3 (Zusammenfassung des DataFrames + infos)
print('#'*150)
print('Aufgabe 1.3')
print(data.info())

# Übung 2.1 (Zeilen 'age', 'occupation' und 'income' ausgeben mit 10 Zeilen)
print('#'*150)
print('Aufgabe 2.1')
new_data = data[['age', 'occupation', 'income']]
print(new_data.head(10))

# Übung 2.2 (Filtern 'income' ist '>50K')
print('#'*150)
print('Aufgabe 2.2')
incomefilter = new_data[(new_data['income'] == '>50K')]
print(incomefilter)

# Übung 2.3 (Filter 'age' ist größer 30 und 'education' ist 'Bachelors')
print('#'*150)
print('Aufgabe 2.3')
a_e_data = data[['age', 'education']]
age_education = a_e_data[(a_e_data['age'] > 30) & (a_e_data['education'] == 'Bachelors')]
print(age_education.head(10))

# Übung 3.1 (neue Spalte hinzufügen: Alter in Jahrzehnten (alter/10))
print('#'*150)
print('Aufgabe 3.1')
data['age_decade'] = data['age']/10
age_decade_data = data[['age', 'age_decade']]
print(age_decade_data.head(10))

# Übung 3.2 (Ersetzte in der 'income'-Spalte die Werte '>50K' und '<=50K' durch 'high' und 'low')
print('#'*150)
print('Aufgabe 3.2')
data = data.replace(['>50K'], 'high')
data = data.replace(['<=50K'], 'low')
print(data.head(10))

# Übung 3.3 (alle Zeilen löschen, indenen die 'occupation' 'Unknown' ist)
print('#'*150)
print('Aufgabe 3.3')
new_data2 = data[['age', 'occupation', 'income']]
new_data2 = new_data2[new_data2['occupation'] != '?']
print(new_data2.head(10))

# Übung 4.1 (zeige die deskriptiven Statistiken für Spalte 'age')
print('#'*150)
print('Aufgabe 4.1')
descriptive_statistic = data.describe()
print(descriptive_statistic)

# Übung 4.2 (Gruppieren und Aggregieren (Berechne den Durchschnitt von 'age', gruppiert nach 'income'))
print('#'*150)
print('Aufgabe 4.2')
gruppiert = data.groupby('income')['age'].mean()
print(gruppiert)

# Übung 4.3 (finde alle einzigartigen Werte in der 'education'-Spalte)
print('#'*150)
print('Aufgabe 4.3')
einzigartig = data['education'].unique()
print(einzigartig)

# Übung 5.1 (Boxplot für 'age', gruppiert nach 'income')
print('#'*150)
print('Aufgabe 5.1')
print('Box-Plot')

fig, ax = plt.subplots(figsize=(10,8))
plt.suptitle('')
data.boxplot(column=['age'], by='income', ax=ax)
plt.show()

# Übung 5.2 (erstelle einen Scatter Plot von 'age' gegen 'hours-per-week', farbkodiert nach 'income')
print('#'*150)
print('Aufgabe 5.2')
print('Scatter-Plot')

plt.figure(figsize = (10, 6))
sns.scatterplot(x='age', y='hours-per-week', hue='income', data=data, palette='coolwarm')
plt.title('Scatter Plot von age gegen hours-per-week')
plt.xlabel('Alter')
plt.ylabel('Stunden pro Woche')

plt.xlim(data['age'].min(), data['age'].max())
plt.ylim(data['hours-per-week'].min(), None) # anstatt None kann auch eine Zahl eingegeben werden

plt.legend(title='Einkommen')
plt.show()


Übung 1: Daten einlesen und inspizieren
1. Daten einlesen: Lade den Datensatz `adult.csv` in einen Pandas DataFrame.
2. Datensatz erkunden: Zeige die ersten 5 und die letzten 5 Zeilen des DataFrames an.
3. Zusammenfassung: Erhalte eine Zusammenfassung des DataFrames mit Informationen zu jeder Spalte (Datentypen, Anzahl der Nicht-Null-Einträge usw.).

Übung 2: Auswahl und Filterung
1. Spalten auswählen: Wähle die Spalten `age`, `occupation` und `income` aus und zeige die ersten 10 Zeilen an.
2. Bedingte Auswahl: Filtere die Daten, um nur die Zeilen anzuzeigen, bei denen das `income` '>50K' ist.
3. Mehrere Bedingungen: Finde alle Einträge, bei denen das `age` größer als 30 ist und das `education` 'Bachelors' ist.

Übung 3: Datenbearbeitung
1. Neue Spalte hinzufügen: Berechne das Alter in Jahrzehnten (alter/10) und füge es als neue Spalte `age_decade` hinzu.
2. Werte ändern: Ersetze in der `income`-Spalte die Werte '>50K' und '<=50K' durch 'high' und 'low' respektive.
3. Zeilen löschen: Entferne alle Zeilen, in denen die `occupation` 'Unknown' ist.

Übung 4: Einfache Datenanalyse
1. Deskriptive Statistiken: Zeige die deskriptiven Statistiken für die `age` Spalte.
2. Gruppieren und Aggregieren: Berechne den Durchschnitt von `age`, gruppiert nach `income`.
3. Einzigartige Werte: Finde alle einzigartigen Werte in der `education`-Spalte.

Übung 5: Visualisierung
1. Boxplot: Erstelle Boxplots für `age`, gruppiert nach `income`.
2. Scatter Plot: Erstelle einen Scatter Plot von `age` gegen `hours-per-week`, farbkodiert nach `income`.