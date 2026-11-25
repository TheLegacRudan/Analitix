note: text na nikakav nacin nije napisan AI-om (osim za stil markdowna) pa se unaprijed ispricavam na kaoticnosti pisanja

# Dizajn i vrednovanje sustava

## Pretpostavke i odluke

Pretpostavka prepoznat osobu ako je dio njenog bounding pravokutnika u unutar zone te ako se zadrzi u zoni 3 sekunde aktivirati alarm. Ako je njen bounding box izasao iz zone ili nestao (kao kod videa s autobusnom stanicom) smatra se da je alarm zavrsen.


Odluke:

- vrijeme se računa po vremenu videa
- površina osobe ne mora cijela biti u zoni: presjek pravokutnika osobe i zone, uključujući kutove i bridove, smatra se ulaskom u zonu.
- svaki nova ID instanca ima svoj alarm, jer je sigurnije ne izostaviti osobu nego spojiti različite osobe u jednu
- omogućen je grace period 0,75 sekunde prije kraja alarma, kako kratki
  propuštaji detekcije ne bi odmah zatvorili aktivan alarm (sto se cesto dešava kod videa s autobusnom stanicom po noci i danu)
- konfiguracije (yaml fileovi) ne naštimavaju se ručno za svaki video nego sustav prvo samplea svjetlost i broj osoba u prvih pet frameova, a onda bira profil `default`, `crowd` ili `night`.
  `crowd` profil koristi BoT-SORT i dulji buffer za praćenje tako da što manje mijenja id-jeve; 
  `night` koristi ByteTrack s nižim pragovima detekcije tako da ne dodjeljuje novi id-jeve za svaku okulziju;
  `default` koristi default tracker, nisam 100% siguran je li nesto mijenjano u odnosu na default yaml konfiguraciju, ali ako je zanemarivo je
- Rezultat: video u cjelosti pohranjen i alarmi pohranjeni u `alarms.json`.

Odbacene alternative:
- **Jedna jedinstvena konfiguracija tracker-a za sve scene** inicijalno tako zamisljeno ali odbijeno jer gužva i noćni uvjeti stvaraju greške na takav način da prilagodjavanje toj greški izaziva greške u ostalim videima
- **Odluka na temelju središta bounding boxa** inicijalno osoba u zoni tek kada je njen centar u zoni, brzo shvaćeno da je takav sustav glup (sklaidste-oosba videa gdje centar nikad ne ulazi u zonu)
- **Odluka da ista osoba moze imati vise alarma** - zbog toga sto osoba moze biti 3 minute u kadru pa 5 minuta van kadra pa opet 3 minute u kadru dovoljno je ostaviti identican id jednoj osobi (ako je to moguce) ali obe instance ulaska u kadar zabiljeziti kao alarm (tj dva razlicita alarma)
- **Odluka da je zona detekcije veća od zone alarma a manja od kompletnog kadra** - iako je potrebno aktivirati alarm samo ako je osoba u zoni alarma, povecavanjem zone detekcije (širenje zone detekcije napravljeno obrnuto reciprocno, mogu detaljnije objasniti uzivo ali to je funkcija "expand_zone") imamo vecu kontrolu id-jeva i bolju detekciju. Alternativa alternativi bi bila detekcija u cijelom kadru (ali alarm samo za zonu) ali to moze biti programski skupo ako je kadar prevlik i ima previse ljudi ili ako je zona premala u odnosu na cijeli kadar (kao kod videa skladiste-osoba


**Odgovori na točke 5-8 su u fileu readme.md**


Pogreške sustava:
- **Lažne osobe u gužvi:** ljudi ili dijelovi scene mogu stvoriti dodatne detekcije i ID-jeve
- **Promjena ili zamjena ID-ja**
- **Propust u slabom svjetlu pri prolasku auta**


## Što nije gotovo i kako bi se riješilo
Nema evaluacija alarma s pouzdanosti/preciznosti
Također sustav očito nije optimalan kompliciranijim scenama

## Korištenje umjetne inteligencije
AI je korišten na način da bi generirao kod za pojedinačne male zadatke koje bi mu ja definirao nakon razrade problema i cilja. Nakon generiranja koda bih proučio kod te ga testirao ukoliko je sve okej nastavio bih korištenjem AI razvijati druge komponente programa.
Nikad nije baš korišten za generiranje velike količine koda odjednom zbog moje želje za većom kontrolom nad projektom te boljim shvaćanjem.
AI je korišten u potupnosti za pisanje i implementaciju testova.

Ai je pogrijesio u detektiranju osoba na način da je gledao samo sredinu objekta, kod ulaska u zonu. Problem je detektiran tako što sam usporedio vremena zabilježena u alamrs.json i vremena na videima (koja san ručno usporedjivao). Nakon sto sam detektirao problem, napisao sam kod koji detektira ulazak ekstremnih tocaka u zonu (kuteva pravokutnika) te sam dao AI-ju da ga rafinira. 
Sada program provjerava sva četiri kuta bounding boxa.
AI je vise puta faliao kod pisanja dobrih parametara za yaml konfiguracije što je ispravljeno trial and error metodom ručno
