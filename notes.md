rijesit - kako se poziva kad koji setting...

bus ljudi - zamjena idjeva
skladiste ljudi - odlicno
skladiste osoba - odlicno
stanica noc - lose, detektira puno razlicitih ljudi (12) iako je samo jedna osoba -> update sad smo na 5 idjeva -> sad smo na 4

ALT SHIFT F
testove runnas ovako  .\venv\Scripts\python.exe -m unittest -v test_alarm_handling



1. za dani video dobijemo listu alarma u alarms.json, dodat nekakvu pouzdanost? - dodat screenshot alarma, ili da video na kraju hajlajta one koji su duze od 3 sec u kadru crvenon bojon to bi bilo dobro
2. yolo
3. testove napisat? ugh

4\.

&#x09;alarmi dobri skroz za videa skladista bez laznih i propustenih u sekundu

&#x09;za guzvu (zasad) vremena ok, dupliranje idjeva kreira lazne alarme i sl. pogledat

&#x09;za osobu po noci kreiraju se dupli idjevi (ukupno 4) sto rezultira sa 3 alarma s razl. idjevima (umjesto jednog) (napomena, zapravo nije jedan jer ako osoba izadje iz zone i vrati se nazad kreirat ce se novi alarm sa starim idjem i to je ok



5\.

&#x09;loše - da li da na neki nacin grupirat ljude ako su skupa? pretesko vjv i nepotrebno

6\.

&#x09;rijetko propusti osobu noću (pogledaj frejmove specificno) kad nema auta

&#x09;kad ima auta propusti osobu ponekad i dodijeli novi id (total 4) -> poboljšanje od povećavanja match\_thresh sa 0.8 na 0.9 i track\_low\_thresh sa 0.1 na 0.05 -> ne kvari druge rezultate



8\. yolo -  AGPL-3.0 License - https://ultralytics.com/license



9\. 

&#x09;9.1. odluka zasad da svaka osoba ima svoj alarm, odluka ostavit duple idjeve (bolje nego da propusti neku osobu), odluka 3 razl modela

&#x09;9.2. mjerenje dobro ali sustav grijesi u slucaju 5. i 6.

&#x09;9.3 odgovoreno

&#x09;9.4 testovi naprvaljeni - nisu do kraja testirani haha

&#x09;9.5 ai pogrijesio u detektiranju osoba na način da je gledao samo sredinu objekta, kod ulaska u zonu u izlaska;  detektirao problem, napisao kod koji detektira ulazak ekstremnih tocaka u zonu (kuteva pravokutnika), ai vise puta falija kod pisanja dobrih parametara za pracenje razlicitih situacija -> rijesija trial and error metodom

