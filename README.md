
## Instalacija

Kad se klonira repo, u rootu runnat jednu po jednu naredbu:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install opencv-python numpy ultralytics
.\.venv\Scripts\python.exe -m pip install "lap>=0.5.12"
```

## Pokretanje

Nakon toga program se runna naredbom, imena videa su `skladiste-ljudi` `skladiste-osoba` `bus-ljudi` `stanica-noc`:

```powershell
python detect_video.py --show --video "skladiste-ljudi"
```

Program se također moze runnati iz VS codea sa Ctrl+F5 i onda automatski pokreće video definiran u .vscode/launch.json fileu

- Videi i njihova imena su u videos folderu
- Rezultat je u output folderu (koji ce se kreirati)
- Alarmi su u alarms.json fileu

## Testovi

Testovi se runnaju ili iz sidebara u VS Codeu ili naredbom

```powershell
.\venv\Scripts\python.exe -m unittest -v test_alarm_handling
```

## Stavke iz zadatka

1. program daje listu alarma u jsonu za video i zonu te daje video rezultat u kojemu su one osobe koje su aktivirale alarm oznacene crvenim pravokutnikom
2. za detektor koristen yolo11 -  AGPL-3.0 License - https://ultralytics.com/license
3. logika alarma pokrivena u alarm_handlingu.py, testovi napisani za alarm handling (test_alarm_handling.py) i za odabiranje profila trackera tj. konfiguracija
4. mjerenje alarma:

   - skladiste ljudi - (1_warehouse_aisle) - alarmi se ponasaju savrseno - sustav koristi "default.yaml" konfiguraciju koju samo odluci primjeniti na temelju pocetnih frameova
   - skladiste osoba - (2_warehouse_walk) - alarm se ponasa savrseno - sustav koristi "default.yaml" konfig.
   - bus ljudi - (4_bus_station_crowd) - objasnjeno u tocki 5
   - stanica noc - (3_bus_stop_night) - objasnjeno u tocki 6

5. rucno mjerenje tocnosti nema previse smisla s obzirom da alarmi imaju dosta gresaka i dosta zamjene id-jeva unatoc naštimavanja konfiguracije i korištenja drugog modela (definiranog u crowd.yaml). Sustav koristi "crowd.yaml" konfig. Prijašnje stanje (s default modelom) bilo znatno losije - unatoc nastimavanju to je najbolji rezultat koji sam dobio.
6. funkcionira dobro, inicijalno (koristenjem default.yaml) bi detektirao puno razlicitih ljudi (ukupno 12) iako je samo jedna osoba -> koristenjem nastimanog modela ukupno detektira 4, te triggera ukupno 3 alarma (zadnja "osoba" ga ne stigne trigeratti).

   trenutacno rijetko propusti osobu noću (pogledaj frejmove specificno) kad nema auta, kad ima auta propusti osobu ponekad i dodijeli novi id - unatoc nastimavanju to je najbolji rezultat koji sam dobio

7. u zasebnom fileu "dizajn.md"
8. yolo -  AGPL-3.0 License - https://ultralytics.com/license
9. u zasebnom fileu "notes.md"