# Book recommendation

Flowchart
![](./assets/flowchart.drawio.svg)

# Search engine

# Recommendation engine

# Spine_segmentation

Retrieve book information from picture by performing mutiple object detection.

Object detection was preferred to instance segmentation because data vertices annotations are not the best (often rectangle, so it does not make sense to perform both object detection and instance semantic segmentation).

Object detection framework is modified retinanet to acccept rotated bounding boxes

# Training data

### Books / users / ratings

Can be downloaded ![here](https://www.kaggle.com/arashnic/book-recommendation-dataset)

### Book spine
Roughly annotated book spine data (contains ~ 650 datapoint).

Can be downloaded ![here](https://data.4tu.nl/articles/dataset/Data_mannually-labelled_accompanying_the_research_on_segmentation_of_book-spine_images/12688436/1?file=24026006)
