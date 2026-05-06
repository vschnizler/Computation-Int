---
Folders:
  - "[[CI]]"
---
Learning Rate = 0.03
Activation = Tanh
Batch size = 10
Noise = 0
Ratio of training to test data = 50%
# teil a)
-> nur 8 Neuronen in der verdeckten Schicht da max 8 erlaubt

| Parameter                        | Parameter Anzahl | stopped at Epoch: | Test loss | Train loss | Kommentar                                  |
| -------------------------------- | ---------------- | ----------------- | --------- | ---------- | ------------------------------------------ |
| X1, X2                           | 2                | 2.052             | 0.3       | 0.208      |                                            |
| X1², X2²                         | 2                | 2.075             | 0.557     | 0.479      | only got worse                             |
| sin(x1), sin(x2)                 | 2                | 1.153             | 0.407     | 0.400      | stopped bc it was stabilizing              |
| X1\*X2, sin(x1), sin(x2)         | 3                | 700               | 0.266     | 0.114      |                                            |
| X1, X2, X1\*X2                   | 3                | 2042              | 0.285     | 0.132      | test loss slower better than training loss |
| X1, X2, X1², X2²                 | 4                | 2028              | 0.004     | 0.006      |                                            |
| X1, X2, sin(x1), sin(x2)         | 4                | 2037              | 0.037     | 0.009      |                                            |
| X1, X2, sin(x1), sin(x2), x1\*x2 | 5                | 390               | 0.036     | 0.018      |                                            |
| X1, X2, X1², X2², X1\*X2         | 5                | 376               | 0.027     | 0.032      |                                            |
| alle                             | 7                | 1482              | 0.047     | 0.006      | Overfitting                                |
# teil b)

| Parameter Anzahl | stopped at Epoch: | Test loss | Train loss |                         |
| ---------------- | ----------------- | --------- | ---------- | ----------------------- |
| 2                | 352               | 0.483     | 0.453      | improvement slowed down |
| 3                | 887               | 0.232     | 0.149      | improvement slowed down |
| 4                | 1112              | 0.013     | 0.008      |                         |
| 5                | 2012              | 0.135     | 0.027      | overfitting             |
 **Kommentar:** je nach Problemstellung eignen sich bestimmte Parameter mehr als andere. Zu wenige Parameter führen zu underfitting, zu viele zu overfitting. Am schnellsten und mit besten Performance sind es 5 Parameter bei a) und 4 Parameter bei b). Bei 7 bzw 5 Parameter sieht man wie der Train Loss sehr klein wird, das test loss aber vergleichsweise groß bleibt.
 Allgemein sorgen mehrere Schichten schneller für gute Ergebnisse und benötigen weniger Parameter.