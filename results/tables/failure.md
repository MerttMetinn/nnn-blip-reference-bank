# Failure analysis (OUR MEASUREMENTS, seed 0, COCO Karpathy test)

## 1. D and E at query level

| Case | Queries | Share |
|---|---|---|
| Both correct | 8495 | 33.97% |
| Only D correct | 990 | 3.96% |
| Only E correct | 837 | 3.35% |
| Both wrong | 14688 | 58.73% |

### 10 random examples: Only D correct (lost by E)

| Query | Correct image | E's top-1 image (its first caption) | Rank of correct image under E |
|---|---|---|---|
| A black computer mouse sitting next to a keyboard | 487774 | 455299: There are two keyboards and a desk top computer an also a mouse that is lit up with blue lights. | 3 |
| Two giraffes are standing behind a wood fence | 472557 | 52386: A giraffe grazing in pen next to a tree trunk. | 2 |
| two zebras are standing next to each other in their pen | 367329 | 176863: The two zebra are walking in a single file line. | 2 |
| a woman with her hands on the glass looking at donuts | 296236 | 170015: Three ladies working in a confectionery with baked goods behind them | 2 |
| A dinner table with three plates of gourmet hamburgers. | 161222 | 435270: A tray topped with breakfast muffin sandwiches on a table . | 2 |
| A dog is sitting on an unmade bed with pillows. | 22192 | 347650: A dog is laying in a chair in front of a book shelf. | 2 |
| a bunch of people are doing tricks on a single motorcycle | 10104 | 339022: The two men are riding their motorcycles in the parking lot. | 4 |
| Small black object sitting on the inside of a toilet bowl. | 182056 | 195790: A black toilet sitting next to a white bath tub. | 2 |
| A traffic sign with bullet holes in it near a wooded area. | 94025 | 88652: A picture of a broken down stop sign. | 3 |
| A fire hydrant on the side of the road outside a business. | 35105 | 327149: a fire hydrant on a sidewalk near a street | 4 |

### 10 random examples: Only E correct (gained by E)

| Query | Correct image | E's top-1 image (its first caption) | Rank of correct image under E |
|---|---|---|---|
| An airport with large jetliners and a bus traveling on a tarmac. | 18975 | 18975: An airport with large jetliners and a bus traveling on a tarmac. | 1 |
| A man and woman smiling while in a forest. | 389480 | 389480: Two people that are posing for a picture. | 1 |
| A boy on a snowboard at the top of a hill. | 1700 | 1700: A man riding a snow board on top of a snow covered slope. | 1 |
| People shopping at a farmer's market for fresh food. | 465862 | 465862: A woman standing in an isle at a farmers market. | 1 |
| A solitary red umbrella on an empty beach. | 490337 | 490337: Bright red umbrella open on the sand of a beach. | 1 |
| Inside of a bathroom with a cat resting on the toilet. | 219567 | 219567: a white toilet with a cat on top of it | 1 |
| Two religious statues sit on a table at an entrance. | 542325 | 542325: A table with two statues sitting under a picture. | 1 |
| People watching a business presentation on a board. | 165225 | 165225: Room of people sitting at long tables with an overhead projector in front. | 1 |
| a man with a moustache is wearing a tie and has his arms crossed | 443689 | 443689: A man with his arms crossing wearing a shirt and tie. | 1 |
| A black and white image of some buildings and a street light. | 322824 | 322824: A one way sign that is on a pole. | 1 |

## 2. The 10 images most often ranked first under E

Each image has about 5 correct captions; a count far above 5 marks a hub.

| Image (cocoid) | First human caption | A | D | E |
|---|---|---|---|---|
| 578740 | An overhead view of a railroad track with a train passing by | 4 | 3 | 48 |
| 434510 | Surfer riding breaking intense wave all the way to shore | 4 | 16 | 44 |
| 299734 | Two guys are playing Frisbee in the park. | 10 | 13 | 42 |
| 56396 | A surfer is moving through a small wave. | 0 | 18 | 40 |
| 172265 | A herd of zebra standing around a tall wooden tree. | 0 | 2 | 40 |
| 557501 | A bathroom is very colorful with blue yellow and red. | 7 | 21 | 35 |
| 267408 | A few people looking at a television that's next to a laptop. | 3 | 6 | 35 |
| 532211 | A bot with a large drink cup jumping his skateboard. | 135 | 20 | 35 |
| 538122 | A skateboarder is performing an aerial manuever during a competition. | 30 | 7 | 34 |
| 300655 | A man in a yellow shirt stands in a dirt circle. | 48 | 10 | 31 |

Arm A: images never ranked first 1268, images ranked first for more than 20 queries 214.

Arm D: images never ranked first 288, images ranked first for more than 20 queries 10.

Arm E: images never ranked first 379, images ranked first for more than 20 queries 42.

## 3. Caption text in the bank (same 22,657 images, 1 caption per image)

| Measure | Human (arm D) | BLIP (arm E) |
|---|---|---|
| Captions | 22657 | 22657 |
| Distinct captions | 22488 | 13227 |
| Captions identical to another image's | 284 | 11879 |
| Mean words per caption | 10.44 | 8.06 |
| Vocabulary size | 8146 | 2018 |

### 10 most frequent BLIP captions

| Caption | Images |
|---|---|
| a train on the tracks | 249 |
| a man playing tennis on a tennis court | 130 |
| a white plate with food on it | 93 |
| a man doing a trick on a skateboard | 87 |
| the gi is brown and white | 85 |
| the sky is blue | 79 |
| a baseball player swinging at a ball during a game | 74 |
| a woman playing tennis on a tennis court | 74 |
| the sky is cloudy | 73 |
| a man riding a wave on top of a surfboard | 68 |
