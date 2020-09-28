# Imitation Learning; Optimal Multiple Path Search Using VAIL

## How to

The Customized environment and actions

![environment actions](./img/girdworld-AE.PNG)<br>


***environment.py*** : Currently, the customized Grid-World of the 20x20 pixel window is configured.

***Expert dataset 1,2*** : Examples of configuring expert dataset with the pickle module

***expert_generator.py*** : You can use this file to create expert data.

***main.py*** : You can run this program by running ***main.py***.

## Result
### two obstacles - 10 x 10 GridWorld
You should need expert data to find approximately 50 shortest paths.

**150 epoch**

![two obstacle-150](./img/2obstacle-150epoch.PNG)<br>

**500 epoch**

![two obstacle-500](./img/2obstacle-500epoch.PNG)<br>


### four obstacles - easy path
You should need expert data to find approximately 200 shortest paths.

**300 epoch**

![four obstacle0-300](./img/epoch-300.png)<br>

**500 epoch**

![four obstacle0-500](./img/epoch-500.PNG)<br>

**700 epoch**

![four obstacle0-700](./img/epoch-500.PNG)<br>

**900 epoch**

![four obstacle0-900](./img/epoch-900.PNG)<br>

**1000 epoch**

![four obstacle0-1000](./img/epoch-1000.PNG)<br>


### four obstacles - difficult path
You should need expert data to find approximately 400-500 shortest paths.

**700 epoch**

![four obstacle1-700](./img/4opstacle2_epoch-700.png)<br>

**900 epoch**

![four obstacle1-900](./img/4opstacle2_epoch-900.png)<br>

**1000 epoch**

![four obstacle1-1000](./img/4opstacle2_epoch-900.png)<br>


## Related papers


## References


