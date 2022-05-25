# Data analysis - Bike rentals in Edinburgh

### Streamlit aplikace
Výstup projektu je bez spouštění kódu k nahlédutí v rámci Streamlit aplikace: [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/tomas-weber1994/pandas_project_engeto_data_academy/main/App_streamlit.py)

Po spuštění aplikace je pro správné zobrazení výstupů třeba kliknout na pravý horní roh - Settings - zaškrtout Wide mode.

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

##### Explorace dat a deskriptivní statistiky
Většinu času jsem pracoval v rámci Jupyter Notebooku, později jsem obsah jednotlivých buněk nakopíroval do vývojářského prostředí Visual Studio Code a dodal odpovídající kód nutný k funkčnosti aplikace Streamlit. Výsledný skript ve vscode nebyl původně v plánu, prospělo by mu zpřehlednění (např. za pomoci definování vlastních funkcí, kód se občas opakuje s mírnými změnami).

Úplně na začátku bylo třeba načíst data ze souboru "edinburgh_bikes.csv" (všechny datasety jsou dostupné v repozitáři). Jedná se o data obsahující informace o výpůjčkách kol ve městě Edinburgh (zejména jméno počáteční a koncové stanice, jejich souřadnice a výpujční doba).
Data jsem neprve vyčistil - některé sloupce bylo třeba převést na odpovídající datový typ, zkontroloval jsem nulové hodnoty (zjištěny pouze v rámci nepodstatných popisných sloupců), z dat jsem následně vybral pouze relevantní sloupce související se zadáním projektu, ty pak později vhodně přejmenoval. Chyby v datech jsem nepozoroval. 

Do listů jsem uložil jména všech aktivních a neaktivních výpujčních stanic. Jako neaktivní stanici jsem si definoval tu, ve které nedošlo k žádné výpůjčce (a vrácení) v posledním měsíci. V rámci deskriptivní části projektu jsem pak pracoval pouze s daty očištěnými o neaktivní stanice. Spočítal jsem celkový počet zápůjček a vrácení kol na každou aktivní stanici (za sledované období 2018 - 2021, ale i průměrně za den). Následně jsem identifikoval stanice, na kterých dochází k převisu půjčených a vrácených kol. Tyto údaje jsem vizualizoval v rámci plotly a matplotlibu. Údaje o výpůjčkách a vrácení lze filtrovat pomocí jména stanice v rámci aplikace Streamlit.

Následně jsem na základě souřadnic jednotlivých stanic vytvořil matici vzájemných vzdáleností v km. Také jsem ke každé stanici našel jinou, která je jí nejblíže. Na základě vzdálenostní matice je možné v rámci Streamlit aplikace spočítat vzdálenost mezi libovolnými stanicemi. Ke shlédnutí je také mapa obsahující umístění všech stanic v Edinburghu vč. údajů o jejich výpůjčkách za den (je patrný vliv umístění v centru a na předměstí).

K zobrazení rozložení četností výpůjčních dob kol jsem využil histogram v rámci knihovny matplotlib. V datech byly nicméně extrémní hodnoty (histogram měl velmi vysoký rozsah hodnot a nebylo možné jej rozumným způsobem zobrazit), proto jsem zobrazil v rámci histogramu pouze hodnoty očištěné o extrémy, ty jsem si definoval jako hodnoty větší nebo menší než tři směrodatné odchylky od průměru (jedná se o např. výpůjční dobu třiceti dní, která výrazně vybočovala). I po očištění dat lze na výsledném histogramu vidět, že většina výpůjček netrvá ani hodinu, rozložení četností výpujčních dob je zprava zešikmené.

##### Analýza poptávky
Poté jsem se zabýval tím, které faktory mohou mít dopad na četnost výpůjček kol. Nejprve jsem v knihovně altair (jedná se o moji oblíbenou knihovnu) zobrazil četnosti výpůjčkek po jednotlivých měsících v rámci sledovaného období (zřetelně lze vidět dopad sezónnosti na výpůjčky). Data byla zapotřebí samozřejmě neprve vhodně upravit a seskupit. Stejný graf je k nahlédnutí také ve Streamlit, přičemž lze opět filtrovat jednotlivé stanice podle jména a srovnávat je tak mezi sebou. Doplňkově jsem data následně seskupil dle jednotlivých měsíců v roce a hodnoty zprůměroval, graf je nicméně velmi podobný. 

Vzhledem k tomu, že data o výpůjčkách kol pochází z období 2018 - 2021, napadlo mě posoudit vliv onemocnění Covid-19 na výpůjčky kol. Údaje o Covid-19 jsou dostupné v tabulce "covid19_UK.csv" (data vychází z tabulky celosvětových covidových údajů, data jsem pomocí SQL pouze vyfiltroval podle země; v tomto případě UK). Bohužel jsem neměl k dispozici data pouze z města Edinburgh, vycházím tak z celostátních údajů. Tabulku jsem spojil s původním datasetem, odstranil jsem chyby v hodnotách (záporný počet nakažených) a vytvořil korelační matici - patrná je negativní souvislost četností výpůjček kol a pozitivních testů (potažmo úmrtí), přičemž korelace je středně silná. Tento vztah jsem zobrazil za pomoci scatterplotu v rámci knihovny altair. Graf jsem proložil regresní přímkou, v tomto případě datům odpovídala exponenciální funkce.

Poté jsem zkoumal, jaká je souvislost mezi počasím a četností výpůjček kol. Tabulku "edinburgh_weather.csv" jsem proto na základě konkrétního data propojil s původním datasetem vycházejícím z "edinburgh_bikes.csv". Data bylo třeba následně opět vyčistit, převést na správné datové typy, přejmenovat sloupce i hodnoty atp. Následně jsem opět vytvořil korelační matici, ze které vyplývá středně silná pozitivní souvislost mezi teplotou a četností výpůjček kol, dále také menší (ale nezanedbatelné) záporné souvislosti mezi vlhkostí vzduchu, sílou větru a četností výpůjček kol. I v tomto případě jsem data vizualizoval za pomoci scatterplotu.

Vhodnou úpravou dat a za pomoci jednoduchého sloupcového grafu jsem zjistil, že o víkendu se kola půjčují v průměru o 14 % více než přes týden. Na základě těchto uvedených souvislostí jsem se pokusil vytvořit predikci četností půjčených kol. 

##### Predikce počtu výpůjček
K predikci četností kol z výše uvedeného jsem musel všechny tři tabulky (údaje o výpůjčkách, počasí a Covid-19) nejprve spojit dohromady - zde již je limitem výraznější redukce dat vzhledem k tomu, že jednotlivé tabulky nevychází ze zcela stejných sledovaných období. Pro zajímavost jsem si opět vygeneroval korelační matici, v rámci které mě zaujala zejména vyšší korelace mezi teplotou a četností půjčených kol (v tomto případě se jedná o období spojené s Covid-19). Mojí hypotézou je, že souvislost mezi teplotou a četností půjčených kol je během covidu vyšší vzhledem k tomu, že teplota (respektive období v roce) také souvisí s počtem pozitivních testů, potažmo úmrtími (což má dopad na výpůjčky kol).

Predikce vychází z modulu sklearn, pracuji s tzv. testovací a trénovací částí datasetu. Do predikčního modelu jsem zahrnul pouze ty sloupce, které přispěly k menšímu rozptylu chyb - jedná se o počet pozitivních testů v daném dni, počet úmrtí na onemocnění Covid-19, teplotu, vlhkost vzduchu, sílu větru, velikost srážek, konkrétní měsíc v roce a rozlišení, zda se jedná o den v týdnu nebo víkend. V rámci Streamlit aplikace si uživatel volí jednotlivé parametry, aplikace následně odhaduje počet vypůjčených kol v Edinburghu za den a za vybraných parametrů. V případě zadání kombinace nepříznivých parametrů (např. velmi vysoký počet pozitivních testů, nízká teplota a vybrání zimního měsíce), docházelo k tomu, že odhad půjčených kol byl záporný. Tuto skutečnost jsem se snažil ošetřit v kódu, který běží na pozadí Streamlit aplikace.

