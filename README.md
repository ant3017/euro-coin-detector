**Euro Coin Detector**
========================


Introduction
------------------------
This project aims to develop a [Euro Coin Detector][4] that can locate and recognize euro coins from natural images and classify them according to their coin denomination and tell their values. 

This project consists of 4 components:  

1) [Euro Coin Classifier](https://github.com/chen-yumin/euro-coin-classifier)  
    The classifier is developed using *Artificial Intelligence* and *Machine Learning* technologies. The classifier uses many images of the euro coin series of each denomination to describe each euro coin type's attributes, such as their shape, size, color, patterns, etc. This is used to generalize the euro coins so later the classifier could be used to determine whether an arbitrary object is a certain denomination of euro coin.  
    
2) [Euro Coin Detector](https://github.com/chen-yumin/euro-coin-detector)  
    This program uses *Image Processing* and *Computer Vision* technologies to recognize the euro coins from natural images. Statistical calculations are used in the euro coin detection algorithm, with the previous trained Euro Coin Classifier, to determine the probabilities of the object classification.  
    
3) [Euro Coin Detection Service API](https://github.com/chen-yumin/euro-coin-detector-server)  
    A web-based API service to allow other developers to easily use this euro coin detection technology for their own projects.  
    
4) [Euro Coin Detection Demo Mobile App](https://github.com/chen-yumin/euro-coin-detector-client)  
	A simple Apache Cordova based cross-platform mobile app developed using AngularJS to demonstrate how this works in mobile devices.  


Results
------------------------
| Original | Processed | Result |
| :---: | :---: | :---: |
| ![Original](demo/original.jpg) | ![Processed](demo/reconstructed-image.jpg) | ![Result](demo/segmented-results.jpg) |

This is the results from the above image.  

|             | Coin 1 | Coin 2   | Coin 3 |
| :---------- | :----: | :------: | :----: |
| Hue         | 151    |  18      | 7      |
| Saturation  | 30     |  127     | 133    |
| Result      | 1 Euro |  20 Cent | 5 Cent |
| Probability | 4.67%  |  70.79%  | 51.29% |


Licensing
------------------------
Please see the file named [LICENSE.md](LICENSE.md).


Author
------------------------
* Chen Yumin  


Contact
------------------------
* Chen Yumin: [*http://chenyumin.com/*][1]
* CharmySoft: [*http://CharmySoft.com/*][2]  
* About: [*http://CharmySoft.com/about*][3]  
* Email: [*hello@chenyumin.com*](mailto:hello@chenyumin.com)  

[1]: http://chenyumin.com/ "Chen Yumin"
[2]: http://www.CharmySoft.com/ "CharmySoft"
[3]: http://www.CharmySoft.com/about "About CharmySoft"
[4]: http://www.CharmySoft.com/app/euro-coin-detector "Euro Coin Detector"