# Rad sustava na Jetsonu sa 16 kamera i bez interneta

## Granica između uređaja i servera

Po ovoj arhitekturi obrada se mora odvijati na Jetson uređaju

Za svaku kameru potrebno je odvojeno držati:

- posljednjih nekoliko frejmova i FPS,
- trenutni alarm state,
- track ID-jeve,
- vrijeme ulaska osobe u zonu,
- zadnje viđeno vrijeme u zoni,
- aktivan alarm i njegovo planirano završetak.

Ovaj direktorij sada ima samo obradu jednog videa, nema paralalnu obradu te ima lokalni `alarms.json`. Za 16 kamera treba nekakva lokalna
SQLite baza ili nesto slicno.

Server bi trebao, kada veza postoji, preuzeti:
- JSON zapise alarma i njihove događaje,
- preslikane/komprimirane snimke
- konfiguracije kamera i alarmnih zona

## Očekivano usko grlo i mjerenje

Najizglednije usko grlo je GPU/vrijeme inferencije više kamera. Jetson mora paralelno
izvoditi dekodiranje, spremanje i prikaz frejmova, GPU YOLO detekciju, tracking,
geometriju zone i pisanje alarma

## Što se događa nakon restarta

Trenutna implementacija `alarm_handling.py` čuva alarme u `states` samo u RAM-u i
JSON zapisuje samo osnovne zapise. Kad se Jetson izgasi ili se proces resetira sve se praktički gubi osim alarms.jsona i vjerojatno broken videa u outputu