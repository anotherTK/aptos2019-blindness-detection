

# EfficientNet

| model           | iters | lr    | acc-1 | loss weighted | test acc |
| --------------- | ----- | ----- | ----- | ------------- | -------- |
| efficientnet-b4 | 30k   | 0.01  | 83.80 |               | 0.652    |
| efficientnet-b4 | 100k  | 0.01  | 82.74 |               | 0.729    |
| efficientnet-b4 | 100k  | 0.001 | 84.19 |               | 0.737    |
| efficientnet-b4 | 100k  | 0.001 | 80.00 | weighted      | 0.621    |
| efficientnet-b7 | 100k  | 0.001 | 72.42 |               |          |
| efficientnet-b7 | 100k  | 0.001 | 72.42 | weighted      | *        |
| efficientnet-b5 | 100k  | 0.001 |       |               |          |