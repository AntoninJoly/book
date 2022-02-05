# Book recommendation

Tools / concepts used in this projects are:
- Elastic stack (elasticsearch & kibana) via docker.
- Fasttext for query classification.
- Nearest neighbours clustering for recommendation cold start.
- Graph neural network for customized recommendations.
- Web scraping and parsing using beautifulsoup for data creation and data augmentation.
- RetinaNet adapted to rotated bboxes (under development).q

Flowchart
![](./assets/flowchart.drawio.svg)

# Search engine

Elasticsearch is a RESTful distributed search engine. It is Java-based and can search and index document files in diverse formats.

**Elasticsearch** is a distributed, RESTful search and analytics engine capable of
solving a growing number of use cases. As the heart of the Elastic Stack, it
centrally stores your data so you can discover the expected and uncover the
unexpected.

**Kibana** lets you visualize your Elasticsearch data and navigate the Elastic Stack, 
so you can do anything from learning why you're getting paged at 2:00 a.m. to 
understanding the impact rain might have on your quarterly numbers.

**Logstash** is a server-side data processing pipeline that ingests data from a
multitude of sources simultaneously, transforms it, and then sends it to your
favorite "stash." 

# Recommendation engine

# Spine_segmentation

Retrieve book information from picture by performing mutiple object detection.

Object detection was preferred to instance segmentation because data vertices annotations are not the best (often rectangle, so it does not make sense to perform both object detection and instance semantic segmentation).

Object detection framework is modified retinanet to acccept rotated bounding boxes

# Training data

### Books / users / ratings

Can be downloaded ![here](https://www.kaggle.com/arashnic/book-recommendation-dataset)

Since I was missing some information, I used web scraping to retrieve book genres. Airflow.

### Book spine
Roughly annotated book spine data (contains ~ 650 datapoint).

Can be downloaded ![here](https://data.4tu.nl/articles/dataset/Data_mannually-labelled_accompanying_the_research_on_segmentation_of_book-spine_images/12688436/1?file=24026006)
