# Data analysis - Bike rentals in Edinburgh

### Streamlit aplikace
Výstup projektu je bez spouštění kódu k nahlédutí v rámci streamlit aplikace: [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/tomas-weber1994/pandas_project_engeto_data_academy/main/App_streamlit.py)
/n
Po spuštění aplikace a správné zobrazení výstupů je třeba kliknout na pravý horní roh - Settings - zaškrtout Wide mode.


### Zadání:

V Edinburghu, stejně jako v dalších městech, funguje systém "bike sharing" - ve městě jsou stanice s koly, člověk si může nějaké půjčit a potom ho vrátit v nějaké další stanici. Problém je, že v některých stanicích se kola pravidelně hromadí a jinde naopak chybí. Provozovatel kol, firma Just Eat Cycles, zadala projekt, jehož cílem je systém zefektivnit.

Coby datový analytik jste součástí týmu, který na projektu pracuje. Vaším úkolem je zpracovat relevantní data a zjistit z nich informace užitečné pro zbytek týmu. Máte k dispozici data o všech výpůjčkách (na ENGETO databázi v tabulce edinburgh_bikes). Proveďte standardní deskriptivní statistiku dat. Také zjistěte minimálně následující informace:

 - identifikujte aktivní a neaktivní stanice
 - identifikujte nejfrekventovanější stanice
 - identifikujte stanice, na kterých se kola hromadí a stanice, kde potenciálně chybí
 - spočítejte vzdálenosti mezi jednotlivými stanicemi
 - jak dlouho trvá jedna výpůjčka? Najděte odlehlé hodnoty, zobrazte histogram

Analýza poptávky:

  - zobrazte vývoj poptávky po půjčování kol v čase
  - identifikujte příčiny výkyvů poptávky
  - zjistěte vliv počasí na poptávku po kolech (údaje o počasí v Edinburghu jsou v tabulce edinburgh_weather)
  - půjčují si lidé kola více o víkendu než během pracovního týdne?
