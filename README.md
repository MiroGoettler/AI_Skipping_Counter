![banner](assets/dashboard_screenshot.JPG)

# AI Skipping Counter

## Author
[@MiroGoettler](https://github.com/MiroGoettler)

## Motivation
In this small project I present a **Python Dashboard for automatic counting of rope skips**. I got the idea for this project because I jump rope for my cardio two to three times a week. As with most training I was trying to improve my performance and started counting the amount of skips and the time of the session. In contrast to counting other exercises like pushups, you have to count a lot faster and to a lot higher numbers. This can be quit annoying and take the fun out of the rope skipping workout.

## Solution
My solution to this problem is a [Plotly Dashboard](https://plotly.com/dash/) with an AI-based Counting System. In this dashboard the user can easily monitore the rope-skipping-session and see metrics like the total amount of skips, the speed of skipping and the time spend. With this Dashboard the user can concentrate on the physical effort and try to improve the performance without worrying about counting the skips.

## Methods
The actual Skip-Counting was implemented with **Human Pose Detection**. The Python Modul Mediapipe by Google provides ready-to-use and real-time methodes for Pose Detection. The method provides a landmark model with 33 pose landmarks. For counting the skips a relative height threshold was determined by observing many videos of rope skipping. If the a landmark from the left or right foot surpasses this height threshold on skip is counted. Because you can rope skip with booth feet at the same time, but can also jump alternating with each foot, the landmark of the foot with the highest Y-coordinate is choosen. In the GIF bellow you can see how the different skipping-styles are counted. For double jumps a second relative height line was determined.

![Skipping Styles](assets/skipping_styles.gif)

For calculating the height of the lines the overall height of the user and the position of the ankles at the start of a skipping-series is used. This means it also needs to be determined at which point the user starts skipping. The best indication for that is the angle of the arms. 

## Tech Stack
- Python (refer to [requirement.txt](https://github.com/MiroGoettler/AK_Skipping_Counter/blob/main/requirements.txt) for all packages used in this project)
    - Flask
    - Dash Plotly 
    - OpenCV
    - [Mediapipe by Google](https://google.github.io/mediapipe/getting_started/python.html) 
