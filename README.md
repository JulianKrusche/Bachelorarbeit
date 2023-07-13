# Bachelorarbeit
Algorithmus zur Ableitung der wesentlichen Geometrieeigenschaften von Carbonfasern

# Übersicht
In diesem GitHub repository ist der Algorithmus hinterlegt, welcher im Rahmen meiner Bachelorarbeit entstanden ist. Das Program ist in der python Version 3.11.3 geschrieben. Im folgenden wird die Anwendung erklärt.


# Dependecies
Folgende Pakete müssen installiert werden. Die Paketversion dient nur zur Sicherheit, falls unter Updates die Funktionen des Programms nicht länger funktionieren.

matplotlib 3.7.1  <br />
tkinter 8.6.12 <br />
numpy 1.24.3 <br />
tabulate 0.9.0 <br />
PIL (Pillow) 9.4.0 <br />
scipy 1.10.1 <br />
shapely 2.0.1 <br />
plyfile 0.7.1 <br />

# Aufbau
Dieses repository enthält drei Varianten des Algorithmus und zwei Dateien mit Funktionen.  <br />
elipse_main_functions.py  <br />
elipse_main_functions_big.py  <br />
Hier sind die Funktionen gespeichert.

fibre_elipse.py ist der vollständige Algotrithmus. Die beiden anderen Varienten enthalten jeweils nur einige Funtionen des Algorithmus.

# Anwendung
Für eine detaillierte Erklärung der Funktionsweise des Programms sei auf meine Bachelorarbeit verwiesen.
1. Herunterladen der python Dateien und der notwenidigen Pakete
2. Öffnen der fibre_ellipse Datei z.B. in VS Code
3. Überprüfen ob alle notwendigen Pakete installiert wurden
4. Starten des Programms
5. Es öffnet sich Datei Explorer Fenster (Anweisung im Terminal beachten)
6. Zum Speicherort des Carbonfaserscans navigieren
7. Datei anklicken und unten rechts auf "öffnen" klicken (Dateityp: .csv oder .ply)
8. Zweites Datei Explorer Fenster: Auswahl Querschnitt [Betonprobe]; es können mehrere Querschnitte ausgewählt werden, dies führt allerdings zu längerer Programmdauer (Dateityp: .png oder .tiff)
9. Drittes Datei Explorer Fenster: Auswahl Querschnitt [Carbpnfaser] (Dateityp: .png oder .tiff)





