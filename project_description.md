Before we dive into details of the implementation, I wanted to give credit to the recruitment team for how interesting and exciting this interview challenge has been. I have been through a fair share of hackerrank interviews and can firmly state that this challenge (unlike hackerrank ones) encouraged me to learn new concepts and actually test my skills at problem solving and not problem memorization. Thank you for that!

### Intro
When initially approaching a challenge, I personally had very little background into object traking, I had the most familiarity with Kalman filters and YOLO computer vision algorithm. After quickly dismissing Kalman filters (since those didn't really fit the application), the next step was jumping to online resources to find how other people have tackled object detection and tracking problems (YOLO felt like an overkill to use from the very beginning).

After some research, Haar Cascades seemed like the go-to approach to tackle similar tasks of face tracking, so that the the selected approach for `V.1` - `V.3`

### V.1 - `/v1.py`
Basic template for KCF object tracker tracking taken off internet using OpenCV. Tracker has to be manually initialized.
* Pros:
    * Very fast at tracking
* Cons:
    * Not automatic, has to be manually initialized
    * Computationally cheap

### V.2 - `/v2.py`
Using Haar cascades to detect the face in the beginning, then tracking using KCF object tracker.
* Pros:
    * Quick and effective
* Cons:
    * Haar Cascades detection for faces is not consistent across different conditions. If the first time the face is not identified or is identified incorrectly, the tracker will fail to track person’s face
    * Still relatively computationally cheap

### V.3 - `/v3.py`
On every frame, detect face and detect eyes using Haar Cascades
* Pros:
    * Acceptable accuracy and fairly quick
* Cons:
    * A lot of false positives (can be avoided adjusting minNeighbours number, but still a feels off)
    * Not consistent in various conditions, often loses sight of the person
    * Still relatively computationally cheap

When I started to feel like I'm hitting the limit of that is possible with Haar Cascades algorithm, the next intuitive step was to attempt to bring in a new technique that is more capable (Haar Cascades gave a lot on false positives, especially in changing environment conditions). Since I had familiarity with YOLO, I decided to attempt using that, especially that it is considered the fastest and most accurate object detection algorithm (within Deep Nets category). 

###V.4 - `/v4.py`
Using YOLO to detect person in the frame on every iteration. Due to the lack of resources on my personal computer, there was no chance I would attempt training even a Tiny-YOLO model, hence the YOLO deep network has been taken from a pretrained model found online
* Pros:
    * Very accurate
* Cons:
    * Very slow
    
After discovering how slow the YOLO actually is (doing some research, revealed that it's generally much slower when it's wrapped in Python and also I have no GPU), which would definitely be impossible with a mobile phone; the next step was to combine the effectiveness of KCF tracking and accuracy of YOLO.

### V.5 - `/v5.py`
Initially detecting person using YOLO; after, using KCF tracker on person’s face. Using the face tracker also detecting eyes using Haar Cascades and identifying if eyes are open or closed using a simple Deep Net. More details and jupyter notebook, where the model was developed and trained can be found in `/eye_classifier`. Model trained on the data from the videos provided and has reached 88% accuracy. 
* Pros:
    * Fairly accurate
    * Fast after initial face has been detected
* Cons:
    * Computationally expensive to run YOLO (like on mobile device running comma.ai software would probably be impossible)

> Note: please use the links below to see demos for V.4 and V.5 of the algorithm. GitHub's limit of 100mb upload for a single file has been exceeded when trying to upload the YOLO model and pretrained weights. If you want to run the model on your computer, use the following link to download YOLO weights and model and place it in `yolo/model_data/` directory. Download Link: https://drive.google.com/drive/folders/1msa1bm-9ki1a3oo57NLaH-64R2gSVinR?usp=sharing


### Video Demos:
1. `./v3.py` -> https://www.youtube.com/watch?v=UVDgBWnpqxM
2. `./v4.py` -> https://youtu.be/FL90mmdCajg
3. `./v5.py` -> https://www.youtube.com/watch?v=bTW1XTbMyuU
