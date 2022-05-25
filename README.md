# Data analysis - Bike rentals in Edinburgh

### Streamlit aplikace
Výstup projektu je bez spouštění kódu k nahlédutí v rámci streamlit aplikace: [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/tomas-weber1994/pandas_project_engeto_data_academy/main/App_streamlit.py)

Po spuštění aplikace a správné zobrazení výstupů je třeba kliknout na pravý horní roh - Settings - zaškrtout Wide mode.

Pro shlédnutí kódu a detailnějších výsledků si doporučuji do počítače z repozitáře stáhnout jupyter notebook ("Engeto_pandas_project_final_notebook.ipynb") a nainstalovat si knihovny ze souboru "requirements.txt".

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

### Postup:

Nejprve bylo třeba načíst data ze souboru "edinburgh_bikes.csv" (všechny datasety jsou dostupné v repozitáři). Jedná se o data obsahující informace o výpůjčkách kol ve městě Edinburgh (zejména jméno počáteční a koncové stanice, jejich souřadnice a výpujční doba).
Data jsem neprve vyčistil - některé sloupce bylo třeba převést na odpovídající datový typ, zkontroloval jsem nulové hodnoty (zjištěny pouze v rámci nepodstatných popisných sloupců), z dat jsem následně vybral pouze relevantní sloupce související se zadáním projektu, ty pak později vhodně přejmenoval. Chyby v datech jsem nepozoroval. 

Do listů jsem uložil jména všech aktivních a neaktivních výpujčních stanic. Jako neaktivní stanici jsem si definoval tu, ve které nedošlo k žádné výpůjčce (a vrácení) v posledním měsíci. V rámci deskriptivní části projektu jsem pak pracoval pouze s daty očištěnými o neaktivní stanice. Spočítal jsem celkový počet zápůjček a vrácení kol na každou aktivní stanici (za sledované období 2018 - 2021, ale i průměrně za den). Následně jsem identifikoval stanice, na kterých dochází k převisu půjčených a vrácených kol.Tyto údaje jsem vizualizoval v rámci plotly a matplotlibu.


