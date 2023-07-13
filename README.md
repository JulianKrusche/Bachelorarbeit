# Bachelorarbeit: Bildgebende Verfahren zur Modellierung von Carbonbeton
Algorithmus zur Ableitung der wesentlichen Geometrieeigenschaften von Carbonfasern

# Übersicht
In diesem GitHub repository ist der Algorithmus hinterlegt, welcher im Rahmen meiner Bachelorarbeit entstanden ist. Das Program ist in der python Version 3.11.3 geschrieben. Im folgenden wird die Anwendung erklärt.

# Dependencies
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
2. Öffnen der fibre_ellipse.py Datei z.B. in VS Code
3. Überprüfen ob alle notwendigen Pakete installiert wurden
4. Starten des Programms
5. Es öffnet sich Datei Explorer Fenster (Anweisung im Terminal beachten)
6. Zum Speicherort des Carbonfaserscans navigieren (Punktwolke)
7. Datei anklicken und unten rechts auf "öffnen" klicken (Dateityp: .csv oder .ply)
8. Zweites Datei Explorer Fenster: Auswahl Querschnitt [Betonprobe]; es können mehrere Querschnitte ausgewählt werden, dies führt allerdings zu längerer Programmdauer (Dateityp: .png oder .tiff)
9. Drittes Datei Explorer Fenster: Auswahl Querschnitt [Carbonfaser] (Dateityp: .png oder .tiff)<br />
10a.   Falls die Punktwolke mehr als 50 Millionen Punkte enthält, wird eine automatische Reduktion empfohlen (siehe Terminal)<br />
10b.   Falls die größere Carbonfasern nicht achsenparallel ist, wird die folgende Grafik erzeugt; das Programm endet anschließend<br />
![image](https://github.com/JulianKrusche/Bachelorarbeit/assets/74180794/75815fad-4c24-4625-a9b6-f8aeb0e3810c)<br />
10c.   Falls die kleinere Carbonfasern nicht achsenparallel ist, wird eine äquivalente Grafik erzeugt; das Fortführen des Programms kann über das Terminal bestimmt werden<br />
10d.   Falls der Abstand der Punkte zur Regressionsgerade zu hoch ist, wird ebenfalls eine entsprechende Grafik erzeugt; das Fortführen des Programms kann über das Terminal bestimmt werden<br />
11. Im Terminal werden die Richtungen der Fasern angegeben. Anschließend werden die gesamten Ergebnisse ausgegeben<br />
    ![image](https://github.com/JulianKrusche/Bachelorarbeit/assets/74180794/837f276d-7eb0-4f17-b46a-a0169c3fd571)<br />
12. Zur Kontrolle wird eine Grafik mit den elliptischen Zylindern und ein Grafik des Querschnitts mit markierter Ober- und Unterkante erstellt
13. Abschließend kann über den Terminal entschieden werden, ob die Ergebnisse exportiert (.txt oder.csv) werden sollen. Die Datei wird im im selben Ordner gespeichert in der fibre_ellipse.py gespeichert ist.


fibre_ellipse_just_data.py <br />
In dieser Version wird nur die Punktwolke berechnet. Somit werden zu Beginn keine Querschnitte übergeben. Die Ergebnisse sind entsprechend weniger umfangreich. <br />

fibre_ellipse_just_data_no_check.py <br />
In dieser Version fallen zusätzlich die Sicherheitsmaßnahmen raus. Da aus diesem Grund auch die Achsen nicht automatische erkannt werden können, müssen diese manuell eingegeben werden. Diese können in Zeile 28 und 29 angegeben werden. Die Ergebnisse dieser Version können fehlerhaft sein und sollten auf ihre Plausabilität und über die erzeugte Endgrafik kontrolliert werden.

