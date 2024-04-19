# object-tracking-cli

## Podejście (wariant przetwarzania w czasie rzeczywistym)
Problem można podzielić na 3 niezależne części:
1. `(ścieżka do pliku -> klatka)` Pobieranie klatek. W tym przypadku zdecydowałem się na stworzenie osobnego wątku do czytania klatek z dysku i dekodowania. Są to operacje poza Pythonowym mutexem (GIL'em), dlatego oddelegowanie tych zadań do osobnego wątku ma sens. Dzięki temu przetwarzając klatki nie musimy już czekać na te operacje. Za pobieranie klatek i umieszczanie ich w kolejce odpowiada klasa `VideoStream` w `object_tracking_cli/video_streaming`. Wspierany jest jedynie format `mp4`. Chociaż aplikacja prawdopodobnie zadziałałaby również dla innych formatów, ma to na celu uniknięcie błędów - testujemy na `mp4` i wiemy, że wszystko powinno działać. Jeśli przetestowalibyśmy inne formaty możnaby dodać je do listy `SUPPORTED_EXTENSIONS`


2. `(klatka -> bounding box)` Detekcja obiektów. Zdecydowałem się na wykorzystanie modelu YOLOv3n (80 klas COCO), który jest dość szybki. Wykorzystanie biblioteki `Ultralytics` wiele ułatwiło (np. dostęp do dowolnego formatu bounding boxów). Klasa odpowiedzialna za inferencję i zapewnienie odpowiedniego formatu bounding boxów znajduje się w `object_tracking_cli/object_detection/detection.py` (`YOLODetector`).

3. `(bounding box -> śledzony obiekt)` Tracking. Przez naiwny tracker rozumiem tracker, który nie wykorzystuje ani żadnej wewnętrznej reprezentacji ani ruchu. Zakładamy jedynie najprostszą rzecz - że obiekt w kolejnej ramce powinien być blisko ostatniej lokalizacji. Zdecydowałem się tutaj na wydzielenie abstrakcyjnej klasy, która zawiera wszystkie metody, które powinien mieć dowolny tracker. Naiwny tracker dziedziczy po tej klasie i implementuje metodę `update`. 
Tracker może działać w dwóch trybach:
- KD-tree: tutaj wykorzystuję fakt, że dopasowując centroidy trackera do centroidów nowych bounding boxów z reguły powinniśmy szukać wśród najbliższych. Idziemy więc po kolei po nowych bounding boxach i szukamy najbliższego zarejestrowanego centroidu. Mogą zdarzyć się jednak konflikty - wtedy szukamy kolejnego najbliższego, który nie był jeszcze przypisany. W rozsądnych przypadkach ten algorytm powinien działać szybciej, natomiast z minusów jest on wrażliwy na kolejność iteracji (jest w pewnym sensie zachłanny)
- inspirowany https://pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/ . Tutaj szukamy po macierzy odległości wszystkich zarejestrowanych centroidów ze wszystkimi centroidami bounding boxów. Mniej efektywny obliczeniowo, ale w trudnych przypadkach może działać lepiej

### Komentarz
- naiwny tracker (w jednym czy drugim wariancie) dobrze może się sprawdzić w prostych przypadkach, ale nie będzie dobrze działał w przypadku gdy obiekty często się mijają. Również nie będzie dobrze działał w przypadku większych okluzji - nie śledzimy ruchu

- brakuje dokumentacji
- nie zdążyłem też przetestować dokładnie trackera z algorytmenm opartym o KD-tree.
- parametrem jest tylko ścieżka pliku i flaga sterująca algorytmem do trackingu (KD-tree czy zwykły)
- CLI jest w dwóch wariantach: 
    - parametry podajemy jako parametr (branch: main)
    - parametry podawane są jako wejście standardowe (branch: cli_std)
- popracowałbym na pewno nad logowaniem

- w przypadku przetwarzania filmu warto zrobić równoległe przetwarzanie w batchach i zapisać przetworzony film. Ze względu na ograniczony czas zdecydowałem się na przetwarzanie sekwencyjne.

## Installation
```bash
pip install .
```

## Test
```bash
pytest .
```

## Usage
```bash
object_tracking test_car.mp4
```