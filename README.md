# Advanced Datamining
Naam: Lisa Hu<br>
Docent: Dave Langers (LADR)

## Aantekeningen
| Soorten ML   | Nominaal      | Numeriek          |
|--------------|---------------|-------------------|
| Supervised   | Classificatie | Regressie         |
| Unsupervised | Clustering    | Dimensie reductie |

Machine Learning: Algoritmen die beter gaan presteren naarmate je ze traint op meer data.<br>
Deep Learning: ML-modellen die hierarchisch zijn opgebouwd. Bestaan uit heel veel lagen.

Rosenblatt's perceptron: Classificatie model dat twee klassen die gescheiden kunnen worden door een rechte lijn perfect kan leren classificeren.

$ŷ = sgn ( \sum_{i} w_i \cdot x_i )$<br>
$w = w - (ŷ - y)\cdot x$<br>
$b = b - (ŷ - y)$


| $x_1$ | $x_2$ | $y$ |
|-------|-------|-----|
| 1     | 0     | +1  |
| 0     | 1     | +1  |
| -1    | 0     | -1  |
| 0     | -1    | -1  |

| $x_1$ | $x_2$ | $y$ | $w_1$ | $w_2$ | $b$ | $ŷ$ | $ŷ-y$ |
|-------|-------|-----|-------|-------|-----|-----|-------|
| +1    | 0     | +1  | 0     | 0     | 0   | 0   | -1    |
| 0     | +1    | +1  | +1    | 0     | +1  | +1  | 0     |
| -1    | 0     | -1  | +1    | 0     | +1  | 0   | +1    |

